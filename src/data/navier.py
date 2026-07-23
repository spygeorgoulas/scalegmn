from .base_datasets import BaseDataset
from pathlib import Path
import torch


class NavierGeometryINRDataset(BaseDataset):
    """
    Dataset for Navier geometry INRs stored as .pth files.

    Expected directory structures supported:

    Case A: split folders only
        dataset_path/
            train/
                inr_00000.pth
                inr_00001.pth
            val/
                inr_00010.pth
            test/
                inr_00020.pth

    Case B: split folders with class subfolders
        dataset_path/
            train/
                0/
                    inr_00000.pth
                    inr_00001.pth
                1/
                    inr_00002.pth
            val/
                0/
                    inr_00010.pth
            test/
                0/
                    inr_00020.pth

    Label behavior:
    - If a parent folder under the split is numeric, that folder name is used as the label.
    - Otherwise, a dummy label 0 is returned.

    This is useful when your task is not classification, but the pipeline still expects a label tensor.
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
    ):
        print(
            f"Initializing NavierGeometryINRDataset | "
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
        """
        Returns the absolute path of the INR checkpoint for one sample.
        """
        rel_path = self.dataset[index]
        abs_path = Path(self.dataset_path) / rel_path
        return str(abs_path), None

    def get_label(self, index, state_dict, aux):
        """
        Returns the label for one sample.

        Behavior:
        - If the parent directory name is numeric, use it as label.
        - Otherwise return dummy label 0.

        Examples:
            train/0/inr_00001.pth -> label 0
            train/1/inr_00002.pth -> label 1
            train/inr_00003.pth   -> label 0
        """
        rel_path = Path(self.dataset[index])

        parent_name = rel_path.parent.name

        if parent_name.isdigit():
            label = int(parent_name)
        else:
            label = 0

        return torch.tensor(label, dtype=torch.long)

    def load_dataset(self, split_path=None):
        """
        Loads dataset entries by scanning the filesystem directly.

        Supported:
        - dataset_path/split/*.pth
        - dataset_path/split/<numeric_or_other_subdir>/*.pth

        Returns:
            A sorted list of relative paths w.r.t. dataset_path
        """
        split_dir = Path(self.dataset_path) / self.split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        file_list = []

        # Case 1: direct .pth files inside split folder
        direct_files = sorted(split_dir.glob("*.pth"))
        file_list.extend([str(p.relative_to(self.dataset_path)) for p in direct_files])

        # Case 2: .pth files inside subfolders
        for sub_dir in sorted(split_dir.iterdir()):
            if sub_dir.is_dir():
                sub_files = sorted(sub_dir.glob("*.pth"))
                file_list.extend([str(p.relative_to(self.dataset_path)) for p in sub_files])

        if len(file_list) == 0:
            raise RuntimeError(f"No .pth files found under: {split_dir}")

        print(f"Loaded {len(file_list)} INR files from {split_dir}")
        return file_list