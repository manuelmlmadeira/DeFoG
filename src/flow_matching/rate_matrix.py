import torch
import torch.nn.functional as F

from flow_matching import flow_matching_utils
from flow_matching.utils import dt_p_xt_g_x1, p_xt_g_x1


class RateMatrixDesigner:

    def __init__(self, rdb, rdb_crit, eta, omega, limit_dist):

        self.omega = omega  # target guidance
        self.eta = eta  # stochasticity
        # Different designs of R^db
        self.rdb = rdb
        self.rdb_crit = rdb_crit
        self.limit_dist = limit_dist
        self.num_classes_X = len(self.limit_dist.X)
        self.num_classes_E = len(self.limit_dist.E)

        print(
            f"RateMatrixDesigner: rdb={rdb}, rdb_crit={rdb_crit}, eta={eta}, omega={omega}"
        )

    def compute_graph_rate_matrix(self, t, node_mask, G_t, G_1_pred):

        X_t, E_t = G_t
        X_1_pred, E_1_pred = G_1_pred

        X_t_label = X_t.argmax(-1, keepdim=True)
        E_t_label = E_t.argmax(-1, keepdim=True)
        sampled_G_1 = flow_matching_utils.sample_discrete_features(
            X_1_pred,
            E_1_pred,
            node_mask=node_mask,
        )
        X_1_sampled = sampled_G_1.X
        E_1_sampled = sampled_G_1.E

        dfm_variables = self.compute_dfm_variables(
            t, X_t_label, E_t_label, X_1_sampled, E_1_sampled
        )

        Rstar_t_X, Rstar_t_E = self.compute_Rstar(dfm_variables)

        Rdb_t_X, Rdb_t_E = self.compute_RDB(
            X_t_label,
            E_t_label,
            X_1_pred,
            E_1_pred,
            X_1_sampled,
            E_1_sampled,
            node_mask,
            t,
            dfm_variables,
        )

        Rtg_t_X, Rtg_t_E = self.compute_R_tg(
            X_1_sampled,
            E_1_sampled,
            X_t_label,
            E_t_label,
            dfm_variables,
        )

        # sum to get the final R_t_X and R_t_E
        R_t_X = Rstar_t_X + Rdb_t_X + Rtg_t_X
        R_t_E = Rstar_t_E + Rdb_t_E + Rtg_t_E

        # Stabilize rate matrices
        R_t_X, R_t_E = self.stabilize_rate_matrix(R_t_X, R_t_E, dfm_variables)

        # # DEBUG
        # import pdb; pdb.set_trace()

        return R_t_X, R_t_E

    def compute_dfm_variables(self, t, X_t_label, E_t_label, X_1_sampled, E_1_sampled):

        dt_p_vals_X, dt_p_vals_E = dt_p_xt_g_x1(
            X_1_sampled,
            E_1_sampled,
            self.limit_dist,
        )  #  (bs, n, dx), (bs, n, n, de)

        dt_p_vals_at_Xt = dt_p_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
        dt_p_vals_at_Et = dt_p_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

        pt_vals_X, pt_vals_E = p_xt_g_x1(
            X_1_sampled,
            E_1_sampled,
            t,
            self.limit_dist,
        )  # (bs, n, dx), (bs, n, n, de)

        pt_vals_at_Xt = pt_vals_X.gather(-1, X_t_label).squeeze(-1)  # (bs, n, )
        pt_vals_at_Et = pt_vals_E.gather(-1, E_t_label).squeeze(-1)  # (bs, n, n, )

        Z_t_X = torch.count_nonzero(pt_vals_X, dim=-1)  # (bs, n)
        Z_t_E = torch.count_nonzero(pt_vals_E, dim=-1)  # (bs, n, n)

        dfm_variables = {
            "pt_vals_X": pt_vals_X,
            "pt_vals_E": pt_vals_E,
            "pt_vals_at_Xt": pt_vals_at_Xt,
            "pt_vals_at_Et": pt_vals_at_Et,
            "dt_p_vals_X": dt_p_vals_X,
            "dt_p_vals_E": dt_p_vals_E,
            "dt_p_vals_at_Xt": dt_p_vals_at_Xt,
            "dt_p_vals_at_Et": dt_p_vals_at_Et,
            "Z_t_X": Z_t_X,
            "Z_t_E": Z_t_E,
        }

        return dfm_variables

    def compute_Rstar(self, dfm_variables):

        # Unpack needed variables
        dt_p_vals_X = dfm_variables["dt_p_vals_X"]
        dt_p_vals_E = dfm_variables["dt_p_vals_E"]
        dt_p_vals_at_Xt = dfm_variables["dt_p_vals_at_Xt"]
        dt_p_vals_at_Et = dfm_variables["dt_p_vals_at_Et"]
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]
        Z_t_X = dfm_variables["Z_t_X"]
        Z_t_E = dfm_variables["Z_t_E"]

        # Numerator of R_t^*
        inner_X = dt_p_vals_X - dt_p_vals_at_Xt[:, :, None]
        inner_E = dt_p_vals_E - dt_p_vals_at_Et[:, :, :, None]
        Rstar_t_numer_X = F.relu(inner_X)  # (bs, n, dx)
        Rstar_t_numer_E = F.relu(inner_E)  # (bs, n, n, de)

        # Denominator
        Rstar_t_denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
        Rstar_t_denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)

        # Final R^\star
        Rstar_t_X = Rstar_t_numer_X / Rstar_t_denom_X[:, :, None]  # (bs, n, dx)
        Rstar_t_E = Rstar_t_numer_E / Rstar_t_denom_E[:, :, :, None]  # (B, n, n, de)

        # # Protect against NaN and too large values (DEBUG normalization)
        # Rstar_t_X = torch.nan_to_num(Rstar_t_X, nan=0.0, posinf=0.0, neginf=0.0)
        # Rstar_t_E = torch.nan_to_num(Rstar_t_E, nan=0.0, posinf=0.0, neginf=0.0)
        # Rstar_t_X[Rstar_t_X > 1e5] = 0.0
        # Rstar_t_E[Rstar_t_E > 1e5] = 0.0

        return Rstar_t_X, Rstar_t_E

    def compute_RDB(
        self,
        X_t_label,
        E_t_label,
        X_1_pred,
        E_1_pred,
        X_1_sampled,
        E_1_sampled,
        node_mask,
        t,
        dfm_variables,
    ):
        # unpack needed variables
        pt_vals_X = dfm_variables["pt_vals_X"]
        pt_vals_E = dfm_variables["pt_vals_E"]

        # dimensions
        dx = pt_vals_X.shape[-1]
        de = pt_vals_E.shape[-1]

        # build mask for Rdb
        if self.rdb == "general":
            x_mask = torch.ones_like(pt_vals_X)
            e_mask = torch.ones_like(pt_vals_E)

        elif self.rdb == "marginal":
            x_limit = self.limit_dist.X
            e_limit = self.limit_dist.E

            Xt_marginal = x_limit[X_t_label]
            Et_marginal = e_limit[E_t_label]

            x_mask = x_limit.repeat(X_t_label.shape[0], X_t_label.shape[1], 1)
            e_mask = e_limit.repeat(
                E_t_label.shape[0], E_t_label.shape[1], E_t_label.shape[2], 1
            )

            x_mask = x_mask > Xt_marginal
            e_mask = e_mask > Et_marginal

        elif self.rdb == "column":
            # Get column idx to pick
            if self.rdb_crit == "max_marginal":
                x_column_idxs = self.limit_dist.X.argmax(keepdim=True).expand(
                    X_t_label.shape
                )
                e_column_idxs = self.limit_dist.E.argmax(keepdim=True).expand(
                    E_t_label.shape
                )
            elif self.rdb_crit == "x_t":
                x_column_idxs = X_t_label
                e_column_idxs = E_t_label
            elif self.rdb_crit == "abs_state":
                x_column_idxs = torch.ones_like(X_t_label) * (dx - 1)
                e_column_idxs = torch.ones_like(E_t_label) * (de - 1)
            elif self.rdb_crit == "p_x1_g_xt":
                x_column_idxs = X_1_pred.argmax(dim=-1, keepdim=True)
                e_column_idxs = E_1_pred.argmax(dim=-1, keepdim=True)
            elif self.rdb_crit == "x_1":  # as in paper, uniform
                x_column_idxs = X_1_sampled.unsqueeze(-1)
                e_column_idxs = E_1_sampled.unsqueeze(-1)
            elif self.rdb_crit == "p_xt_g_x1":
                x_column_idxs = pt_vals_X.argmax(dim=-1, keepdim=True)
                e_column_idxs = pt_vals_E.argmax(dim=-1, keepdim=True)
            elif self.rdb_crit == "xhat_t":
                sampled_1_hat = flow_matching_utils.sample_discrete_features(
                    pt_vals_X,
                    pt_vals_E,
                    node_mask=node_mask,
                )
                x_column_idxs = sampled_1_hat.X.unsqueeze(-1)
                e_column_idxs = sampled_1_hat.E.unsqueeze(-1)
            else:
                raise NotImplementedError

            # create mask based on columns picked
            x_mask = F.one_hot(x_column_idxs.squeeze(-1), num_classes=dx)
            x_mask[(x_column_idxs == X_t_label).squeeze(-1)] = 1.0
            e_mask = F.one_hot(e_column_idxs.squeeze(-1), num_classes=de)
            e_mask[(e_column_idxs == E_t_label).squeeze(-1)] = 1.0

        elif self.rdb == "entry":
            if self.rdb_crit == "abs_state":
                # select last index
                x_masked_idx = torch.tensor(
                    dx
                    - 1  # delete -1 for the last index
                    # dx - 1
                ).to(
                    self.device
                )  # leaving this for now, can change later if we want to explore it a bit more
                e_masked_idx = torch.tensor(de - 1).to(self.device)

                x1_idxs = X_1_sampled.unsqueeze(-1)  # (bs, n, 1)
                e1_idxs = E_1_sampled.unsqueeze(-1)  # (bs, n, n, 1)
            if self.rdb_crit == "first":  # here in all datasets it's the argmax
                # select last index
                x_masked_idx = torch.tensor(0).to(
                    self.device
                )  # leaving this for now, can change later if we want to explore it a bit more
                e_masked_idx = torch.tensor(0).to(self.device)

                x1_idxs = X_1_sampled.unsqueeze(-1)  # (bs, n, 1)
                e1_idxs = E_1_sampled.unsqueeze(-1)  # (bs, n, n, 1)
            else:
                raise NotImplementedError

            # create mask based on columns picked
            # bs, n, _ = X_t_label.shape
            # x_mask = torch.zeros((bs, n, dx), device=self.device)  # (bs, n, dx)
            x_mask = torch.zeros_like(pt_vals_X)  # (bs, n, dx)
            xt_in_x1 = (X_t_label == x1_idxs).squeeze(-1)  # (bs, n, 1)
            x_mask[xt_in_x1] = F.one_hot(x_masked_idx, num_classes=dx).float()
            xt_in_masked = (X_t_label == x_masked_idx).squeeze(-1)
            x_mask[xt_in_masked] = F.one_hot(
                x1_idxs.squeeze(-1), num_classes=dx
            ).float()[xt_in_masked]

            # e_mask = torch.zeros((bs, n, n, de), device=self.device)  # (bs, n, dx)
            e_mask = torch.zeros_like(pt_vals_E)
            et_in_e1 = (E_t_label == e1_idxs).squeeze(-1)
            e_mask[et_in_e1] = F.one_hot(e_masked_idx, num_classes=de).float()
            et_in_masked = (E_t_label == e_masked_idx).squeeze(-1)
            e_mask[et_in_masked] = F.one_hot(
                e1_idxs.squeeze(-1), num_classes=de
            ).float()[et_in_masked]

        else:
            raise NotImplementedError(f"Not implemented rdb type: {self.rdb}")

        # stochastic rate matrix
        Rdb_t_X = pt_vals_X * x_mask * self.eta
        Rdb_t_E = pt_vals_E * e_mask * self.eta

        return Rdb_t_X, Rdb_t_E

    def compute_R_tg(
        self,
        X_1_sampled,
        E_1_sampled,
        X_t_label,
        E_t_label,
        dfm_variables,
    ):
        """Target guidance rate matrix"""

        # Unpack needed variables
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]
        Z_t_X = dfm_variables["Z_t_X"]
        Z_t_E = dfm_variables["Z_t_E"]

        # Numerator
        X1_onehot = F.one_hot(X_1_sampled, num_classes=self.num_classes_X).float()
        E1_onehot = F.one_hot(E_1_sampled, num_classes=self.num_classes_E).float()
        mask_X = X_1_sampled.unsqueeze(-1) != X_t_label
        mask_E = E_1_sampled.unsqueeze(-1) != E_t_label
        
        # # pc
        # X1_onehot = X1_onehot - self.limit_dist.X.unsqueeze(0).unsqueeze(0)
        # E1_onehot = E1_onehot - self.limit_dist.E.unsqueeze(0).unsqueeze(0)
        # # import pdb; pdb.set_trace()

        Rtg_t_numer_X = X1_onehot * self.omega * mask_X
        Rtg_t_numer_E = E1_onehot * self.omega * mask_E

        # Denominator
        denom_X = Z_t_X * pt_vals_at_Xt  # (bs, n)
        denom_E = Z_t_E * pt_vals_at_Et  # (bs, n, n)

        # Final R^TG
        Rtg_t_X = Rtg_t_numer_X / denom_X[:, :, None]
        Rtg_t_E = Rtg_t_numer_E / denom_E[:, :, :, None]

        return Rtg_t_X, Rtg_t_E

    def stabilize_rate_matrix(self, R_t_X, R_t_E, dfm_variables):

        # Unpack needed variables
        pt_vals_X = dfm_variables["pt_vals_X"]
        pt_vals_E = dfm_variables["pt_vals_E"]
        pt_vals_at_Xt = dfm_variables["pt_vals_at_Xt"]
        pt_vals_at_Et = dfm_variables["pt_vals_at_Et"]

        # protect to avoid NaN and too large values
        R_t_X = torch.nan_to_num(R_t_X, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_E = torch.nan_to_num(R_t_E, nan=0.0, posinf=0.0, neginf=0.0)
        R_t_X[R_t_X > 1e5] = 0.0
        R_t_E[R_t_E > 1e5] = 0.0

        # Set p(x_t | x_1) = 0 or p(j | x_1) = 0 cases to zero, which need to be applied to Rdb too
        dx = R_t_X.shape[-1]
        de = R_t_E.shape[-1]
        R_t_X[(pt_vals_at_Xt == 0.0)[:, :, None].repeat(1, 1, dx)] = 0.0
        R_t_E[(pt_vals_at_Et == 0.0)[:, :, :, None].repeat(1, 1, 1, de)] = 0.0

        # zero-out certain columns of R, which is implied in the computation of Rdb
        # if the probability of a place is 0, then we should not consider it in the R computation
        R_t_X[pt_vals_X == 0.0] = 0.0
        R_t_E[pt_vals_E == 0.0] = 0.0

        return R_t_X, R_t_E
