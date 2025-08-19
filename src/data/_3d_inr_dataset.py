from .base_datasets import BaseDataset
import os
from pathlib import Path
import torch


class Labeled3DINRDataset(BaseDataset):
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
            image_size=(28, 28, 28),  # Assuming 3D image dimensions
            direction='forward',
            layer_layout=None,
            return_path=False,
            data_format="graph",
            switch_to_canon=True,
    ):
        print(f"Initializing Labeled3DINRDataset with split: {split}, dataset_path: {dataset_path}")
        super().__init__(
            dataset,
            dataset_path,
            split_path,
            split,
            node_pos_embed,
            edge_pos_embed,
            equiv_on_hidden,
            get_first_layer_mask,
            image_size,
            layer_layout,
            direction,
            return_path,
            data_format,
            switch_to_canon
        )

        #if debug:
            #print("Debug mode is enabled. Reducing dataset size for testing.")
            #self.dataset = self.dataset[:16]

    def get_path(self, index):
        """
        Returns the file path for a given dataset index.
        """
        file_path = Path(self.dataset_path) / self.dataset[index]
        #print(f"Getting path for index {index}: {file_path}")
        return str(file_path), None

    def get_label(self, index, state_dict, aux):
        """
        Extracts the label from the file path.
        """
        file_path = self.dataset[index]
        label = file_path.split("/")[-2]
        #print(f"Extracted label for index {index}: {label}")
        label_tensor = torch.tensor(int(label))
        return label_tensor

    def load_dataset(self, split_path=None):
        """
        Loads the dataset directly from the dataset path.
        Assumes that the dataset directory contains subdirectories for each split (e.g., train, val, test).
        Each split directory contains subdirectories named numerically (e.g., 0, 1, 2, ...), and each of those contains files.
        """
        dataset_dir = Path(self.dataset_path) / self.split
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

        file_list = []
        for sub_dir in dataset_dir.iterdir():
            if sub_dir.is_dir():
                files = list(sub_dir.glob("*.pth")) #TODO: h5 to pt
                file_list.extend([str(file.relative_to(self.dataset_path)) for file in files])

        #print(f"Loaded {len(file_list)} entries from dataset directory: {dataset_dir}")
        return file_list