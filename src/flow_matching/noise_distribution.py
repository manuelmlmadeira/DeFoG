import torch

from src import utils


class NoiseDistribution:

    def __init__(self, model_transition, dataset_infos):

        self.x_num_classes = dataset_infos.output_dims["X"]
        self.e_num_classes = dataset_infos.output_dims["E"]
        self.y_num_classes = dataset_infos.output_dims["y"]
        self.x_added_classes = 0
        self.e_added_classes = 0
        self.y_added_classes = 0
        self.transition = model_transition

        if model_transition == "uniform":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

        elif model_transition == "absorbfirst":
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[0] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[0] = 1

        elif model_transition == "argmax":
            node_types = dataset_infos.node_types.float()
            x_marginals = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_marginals = edge_types / torch.sum(edge_types)

            x_max_dim = torch.argmax(x_marginals)
            e_max_dim = torch.argmax(e_marginals)
            x_limit = torch.zeros(self.x_num_classes)
            x_limit[x_max_dim] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[e_max_dim] = 1

        elif model_transition == "absorbing":
            # only add virtual classes when there are several
            if self.x_num_classes > 1:
                # if self.x_num_classes >= 1:
                self.x_num_classes += 1
                self.x_added_classes = 1
            if self.e_num_classes > 1:
                self.e_num_classes += 1
                self.e_added_classes = 1

            x_limit = torch.zeros(self.x_num_classes)
            x_limit[-1] = 1
            e_limit = torch.zeros(self.e_num_classes)
            e_limit[-1] = 1

        elif model_transition == "marginal":

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "edge_marginal":
            x_limit = torch.ones(self.x_num_classes) / self.x_num_classes

            edge_types = dataset_infos.edge_types.float()
            e_limit = edge_types / torch.sum(edge_types)

        elif model_transition == "node_marginal":
            e_limit = torch.ones(self.e_num_classes) / self.e_num_classes

            node_types = dataset_infos.node_types.float()
            x_limit = node_types / torch.sum(node_types)

        else:
            raise ValueError(f"Unknown transition model: {model_transition}")

        y_limit = torch.ones(self.y_num_classes) / self.y_num_classes  # typically dummy
        print(
            f"Limit distribution of the classes | Nodes: {x_limit} | Edges: {e_limit}"
        )
        self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

    def update_input_output_dims(self, input_dims):
        input_dims["X"] += self.x_added_classes
        input_dims["E"] += self.e_added_classes
        input_dims["y"] += self.y_added_classes

    def update_dataset_infos(self, dataset_infos):
        if hasattr(dataset_infos, "atom_decoder"):
            dataset_infos.atom_decoder = (
                dataset_infos.atom_decoder + ["Y"] * self.x_added_classes
            )

    def get_limit_dist(self):
        return self.limit_dist

    def get_noise_dims(self):
        return {
            "X": len(self.limit_dist.X),
            "E": len(self.limit_dist.E),
            "y": len(self.limit_dist.E),
        }

    def ignore_virtual_classes(self, X, E, y=None):
        if self.transition == "absorbing":
            new_X = X[..., : -self.x_added_classes]
            new_E = E[..., : -self.e_added_classes]
            new_y = y[..., : -self.y_added_classes] if y is not None else None
            return new_X, new_E, new_y
        else:
            return X, E, y

    def add_virtual_classes(self, X, E, y=None):
        x_virtual = torch.zeros_like(X[..., :1]).repeat(1, 1, self.x_added_classes)
        new_X = torch.cat([X, x_virtual], dim=-1)

        e_virtual = torch.zeros_like(E[..., :1]).repeat(1, 1, 1, self.e_added_classes)
        new_E = torch.cat([E, e_virtual], dim=-1)

        if y is not None:
            y_virtual = torch.zeros_like(y[..., :1]).repeat(1, self.y_added_classes)
            new_y = torch.cat([y, y_virtual], dim=-1)
        else:
            new_y = None

        return new_X, new_E, new_y
