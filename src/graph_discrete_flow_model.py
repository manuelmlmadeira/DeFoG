import time
import wandb
import os

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.categorical import Categorical

from models.transformer_model import GraphTransformer

from metrics.train_metrics import TrainLossDiscrete
from src import utils
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.time_distorter import TimeDistorter
from flow_matching.rate_matrix import RateMatrixDesigner
from flow_matching.utils import p_xt_g_x1
from flow_matching import flow_matching_utils


class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        test_labels=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        self.model_dtype = torch.float32
        self.conditional = cfg.general.conditional
        self.test_labels = test_labels

        # number of steps used for sampling
        self.sample_T = cfg.sample.sample_steps

        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist
        print("max num nodes: ", len(self.node_dist.prob) - 1)
        print("min num nodes: ", torch.where(self.node_dist.prob > 0)[0][0].item())

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        self.train_loss = TrainLossDiscrete(
            self.cfg.model.lambda_train,
        )

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "sampling_metrics",
            ],
        )

        # logging
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.val_counter = 0
        self.adapt_counter = 0

        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            alpha=1,
            beta=1,
        )

        # rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.cfg.sample.rdb,
            rdb_crit=self.cfg.sample.rdb_crit,
            eta=self.cfg.sample.eta,
            omega=self.cfg.sample.omega,
            limit_dist=self.limit_dist,
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return

        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1

        dense_data, node_mask = utils.to_dense(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}"
        )
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_validation_epoch_start(self) -> None:
        print("Starting validation...")
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        return

    def on_validation_epoch_end(self) -> None:
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=False, save_samples=False, save_visualization=True
            )
            to_log = self.evaluate_samples(
                samples=samples, labels=labels, is_test=False
            )

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        self.print("Finished validation.")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.sampling_metrics.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        return

    def on_test_epoch_end(self) -> None:

        if self.cfg.sample.search:
            print("Starting sampling optimization...")
            self.search_hyperparameters()
        else:
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=True,
                save_samples=self.cfg.general.save_samples,
                save_visualization=True,
            )
            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"test_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

            self.print("Finished testing.")

    def sample(self, is_test, save_samples, save_visualization):

        # Load generated samples if they exist
        if self.cfg.general.generated_path:
            self.print("Loading generated samples...")
            with open(self.cfg.general.generated_path, "rb") as f:
                samples = pickle.load(f)
            # Set labels to None
            labels = [None] * len(samples)
            return samples, None

        # Otherwise, generate new samples
        if is_test:
            samples_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        samples = []
        labels = []
        graph_id = 0
        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{samples_to_generate}",
                end="",
                flush=True,
            )
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            num_chain_steps = min(self.number_chain_steps, self.sample_T)
            cur_samples, cur_labels = self.sample_batch(
                graph_id,
                to_generate,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=num_chain_steps,
                save_visualization=save_visualization,
            )
            samples.extend(cur_samples)
            labels.extend(cur_labels)

            graph_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if save_samples:
            self.print("Saving the generated graphs")

            # saving in txt version
            filename = "graphs.txt"
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            # saving in pkl version
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            print("Generated graphs saved.")

        return samples, labels

    def evaluate_samples(
        self,
        samples,
        labels,
        is_test,
        save_filename="",
    ):
        print("Computing sampling metrics...")

        to_log = {}
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate
        if is_test:
            for i in range(self.cfg.general.num_sample_fold):
                cur_samples = samples[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]
                cur_labels = labels[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]

                cur_to_log = self.sampling_metrics.forward(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,
                    name=f"self.name_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=is_test,
                    local_rank=self.local_rank,
                    labels=cur_labels if self.conditional else None,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_fold{i}_{save_filename}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }
        else:
            to_log = self.sampling_metrics.forward(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=is_test,
                local_rank=self.local_rank,
                labels=labels if self.conditional else None,
            )

        return to_log

    def apply_noise(self, X, E, y, node_mask, t=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = p_xt_g_x1(
            X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
        )

        # step 4 - sample noised data
        sampled_t = flow_matching_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
        save_visualization: bool = True,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.noise_dist.get_limit_dist(), node_mask=node_mask
        )
        if self.conditional:
            if "qm9" in self.cfg.dataset.name:
                y = self.test_labels
                perm = torch.randperm(y.size(0))
                idx = perm[:100]
                condition = y[idx]
                condition = condition.to(self.device)
                z_T.y = condition.repeat([10, 1])[:batch_size, :]
            elif "tls" in self.cfg.dataset.name:
                z_T.y = torch.zeros(batch_size, 1).to(self.device)
                z_T.y[: batch_size // 2] = 1
            else:
                raise NotImplementedError
        X, E, y = z_T.X, z_T.E, z_T.y

        # Init chain storing variables
        assert (E == torch.transpose(E, 1, 2)).all()
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps + 1, keep_chain, E.size(1), E.size(2))
        )
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps + 1, keep_chain))
        chain_time_unit = 1 / number_chain_steps

        # Store initial graph
        if keep_chain > 0:
            sampled_initial = z_T.mask(node_mask, collapse=True)
            chain_X[0] = sampled_initial.X[:keep_chain]
            chain_E[0] = sampled_initial.E[:keep_chain]
            chain_times[0] = torch.zeros((keep_chain))

        for t_int in tqdm(range(0, self.cfg.sample.sample_steps)):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / (self.cfg.sample.sample_steps + 1)
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / (self.cfg.sample.sample_steps + 1)

            # using round for precision
            write_index = int(np.ceil(np.round(s_norm[0].item() / chain_time_unit, 6)))

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:

            # Repeat last frame 10x to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 1 + 10)

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        chain_X, chain_E, _ = self.noise_dist.ignore_virtual_classes(
            chain_X, chain_E, y
        )

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            # Visualize chains
            self.print("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        chain_times[:, i].numpy(),
                    )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules), end="", flush=True
                )
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        # assert (E_t.argmax(-1) < 4).all()
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        # , condition
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        if self.conditional:
            uncond_y = torch.ones_like(y_t, device=self.device) * -1
            noisy_data["y_t"] = uncond_y

            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            R_t_X_uncond, R_t_E_uncond = (
                self.rate_matrix_designer.compute_graph_rate_matrix(
                    t,
                    node_mask,
                    G_t,
                    G_1_pred,
                )
            )

            guidance_weight = self.cfg.general.guidance_weight
            R_t_X = torch.exp(
                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_X + 1e-6) * guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_E + 1e-6) * guidance_weight
            )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def search_hyperparameters(self):
        """
        Grid search for sampling hypeparameters.
        The num_step_list is tunable based on requirements.
        """

        num_step_list = [5, 10, 50, 100, 1000]
        if self.cfg.dataset.name in "qm9":
            # num_step_list = [1, 5, 10, 50, 100, 500]
            num_step_list = [5, 10]
        if self.cfg.dataset.name == "guacamol":  # accelerate
            num_step_list = [50]
        if self.cfg.dataset.name == "moses":  # accelerate
            num_step_list = [50]

        if self.cfg.sample.search == "all":
            results_df = self.search_distortion(num_step_list)
            results_df = self.search_stochasticity(num_step_list)
            results_df = self.search_target_guidance(num_step_list)
        elif self.cfg.sample.search == "distortion":
            results_df = self.search_distortion(num_step_list)
        elif self.cfg.sample.search == "stochasticity":
            results_df = self.search_stochasticity(num_step_list)
        elif self.cfg.sample.search == "target_guidance":
            results_df = self.search_target_guidance(num_step_list)
        else:
            raise NotImplementedError(
                f"Search type {self.cfg.sample.search} not implemented."
            )

        print("Finished searching. Results saved to search_hyperparameters.csv")

    def search_distortion(self, num_step_list):
        """
        Grid search for sampling distortion.
        """
        results_df = pd.DataFrame()
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        # distortion_list = ["identity", "polydec"]

        for num_step in num_step_list:
            for distortor in distortion_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.time_distortion = distortor
                print(
                    f"############# Testing num steps: {num_step}, distortor: {distortor} #############"
                )
                samples, labels = self.sample(
                    is_test=True,
                    save_samples=self.cfg.general.save_samples,
                    save_visualization=False,
                )
                res = self.evaluate_samples(
                    samples=samples, labels=labels, is_test=True
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["distortor"] = distortor
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_distortion.csv")

        # set back to default value
        self.cfg.sample.time_distortion = "identity"

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "distortor"], inplace=True)
        results_df.to_csv(f"search_distortion.csv")

    def search_stochasticity(self, num_step_list):
        """
        Grid search for stochasticity level eta.
        The num_step_list is tunable based on requirements.
        """
        results_df = pd.DataFrame()
        eta_list = [0.0, 5, 10, 25, 50, 100, 200]
        # eta_list = [5, 10]
        for num_step in num_step_list:
            for eta in eta_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.eta = eta
                print(
                    f"############# Testing num steps: {num_step}, eta: {eta} #############"
                )
                samples, labels = self.sample(
                    is_test=True,
                    save_samples=self.cfg.general.save_samples,
                    save_visualization=False,
                )
                res = self.evaluate_samples(
                    samples=samples, labels=labels, is_test=True
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["eta"] = eta
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_stochasticity.csv")

        # set back to default value
        self.cfg.sample.eta = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "eta"], inplace=True)
        results_df.to_csv(f"search_stochasticity.csv")

    def search_target_guidance(self, num_step_list):
        """
        Grid search for target guidance omega.
        The num_step_list is tunable based on requirements.
        """
        results_df = pd.DataFrame()
        omega_list = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            1.0,
            2.0,
        ]  # tunable based on requirements
        # omega_list = [0.5, 0.01]  # tunable based on requirements

        for num_step in num_step_list:
            for omega in omega_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.omega = omega
                print(
                    f"############# Testing num steps: {num_step}, omega: {omega} #############"
                )
                samples, labels = self.sample(
                    is_test=True,
                    save_samples=self.cfg.general.save_samples,
                    save_visualization=False,
                )
                res = self.evaluate_samples(
                    samples=samples, labels=labels, is_test=True
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["omega"] = omega
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_target_guidance.csv")

        # set back to default value
        self.cfg.sample.omega = 0.0

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "omega"], inplace=True)
        results_df.to_csv(f"search_target_guidance.csv")
