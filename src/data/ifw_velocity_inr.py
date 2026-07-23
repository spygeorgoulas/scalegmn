from .base_datasets import BaseDataset, Batch
from .data_utils import to_pyg_batch, get_node_types, get_edge_types, nn_to_edge_index
from pathlib import Path
import torch


class IFWVelocityINRDataset(BaseDataset):
    """
    Dataset for GRaM / warped-IFW velocity INRs stored as .pth files.

    These INRs correspond to the task:
        f(x, y, z, t) -> (u, v, w)
    """

    def __init__(
        self,
        dataset,
        dataset_path,
        split_path=None,
        debug=False,
        split="train",
        node_pos_embed=False,
        edge_pos_embed=False,
        equiv_on_hidden=False,
        get_first_layer_mask=False,
        image_size=(129, 129),
        direction="forward",
        layer_layout=None,
        return_path=False,
        data_format="graph",
        switch_to_canon=False,
        **kwargs,
    ):
        print(
            f"Initializing IFWVelocityINRDataset | "
            f"split={split} | dataset_path={dataset_path}"
        )

        super().__init__(
            dataset=dataset,
            dataset_path=dataset_path,
            split_path=split_path,
            split=split,
            node_pos_embed=node_pos_embed,
            edge_pos_embed=edge_pos_embed,
            equiv_on_hidden=equiv_on_hidden,
            get_first_layer_mask=get_first_layer_mask,
            image_size=image_size,
            layer_layout=layer_layout,
            direction=direction,
            return_path=return_path,
            data_format=data_format,
            switch_to_canon=switch_to_canon,
        )

        if debug:
            self.dataset = self.dataset[:16]
            print("Debug mode enabled: using first 16 samples only.")

    def get_path(self, index):
        rel_path = self.dataset[index]
        abs_path = Path(self.dataset_path) / rel_path
        return str(abs_path), None

    def get_label(self, index, state_dict, aux):
        rel_path = Path(self.dataset[index])
        parent_name = rel_path.parent.name

        if parent_name.isdigit():
            label = int(parent_name)
        else:
            label = 0

        return torch.tensor(label, dtype=torch.long)

    def load_dataset(self, split_path=None):
        split_dir = Path(self.dataset_path) / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        file_list = []

        direct_files = sorted(split_dir.glob("*.pth"))
        file_list.extend([str(p.relative_to(self.dataset_path)) for p in direct_files])

        for sub_dir in sorted(split_dir.iterdir()):
            if sub_dir.is_dir():
                sub_files = sorted(sub_dir.glob("*.pth"))
                file_list.extend([str(p.relative_to(self.dataset_path)) for p in sub_files])

        if len(file_list) == 0:
            raise RuntimeError(f"No .pth files found under: {split_dir}")

        print(f"Loaded {len(file_list)} INR files from {split_dir}")
        return file_list

    def __getitem__(self, index):
        """
        Custom override so we can:
        1. load the path
        2. convert weights/biases to the format expected by BaseDataset.batch_to_graphs
        3. still return the path for matching with the target npz file
        4. return Batch(weights, biases, label) for the training script
        """
        path, aux = self.get_path(index)
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = self.get_label(index, state_dict, aux)

        # IMPORTANT:
        # Convert standard PyTorch Linear tensors:
        #   weight: (out, in) -> (in, out, 1)
        #   bias:   (out,)    -> (out, 1)
        weights = tuple(
            v.permute(1, 0).unsqueeze(-1)
            for k, v in state_dict.items()
            if "weight" in k
        )
        biases = tuple(
            v.unsqueeze(-1)
            for k, v in state_dict.items()
            if "bias" in k
        )

        if self.data_format == "wb":
            w_b = Batch(weights=weights, biases=biases, label=label)
            return w_b, path

        # Convert to graph
        node_features, edge_features = self.batch_to_graphs(weights, biases)

        batch, _ = to_pyg_batch(
            node_features,
            edge_features,
            self.edge_index,
            node2type=self.node2type if self.node_pos_embed else None,
            edge2type=self.edge2type if self.edge_pos_embed else None,
            direction=self.direction,
            label=label,
            hidden_nodes=self.hidden_nodes if self.equiv_on_hidden else None,
            first_layer_nodes=self.first_layer_nodes if self.get_first_layer_mask else None,
        )

        # Return exactly what the standalone training script needs
        w_b = Batch(weights=weights, biases=biases, label=label)
        return batch, w_b, path