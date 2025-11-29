import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os

class LFWDataset(Dataset):
    def __init__(
        self,
        faces_folder,
        smiling_labels_file,
        non_smiling_labels_file,
        transform=None
    ):
        self.faces_folder = faces_folder
        self.transform = transform

        # TODO: Read the smiling faces file and extract filenames
        # Hint: Open the file, read lines, strip whitespace, and replace .jpg with .ppm
        # Filter out lines that end with 'listt.txt'
        smiling_files = [] # type: list
        with open(smiling_labels_file, "r") as f:
            for line in f:
                name = line.strip()
                if not name or name.endswith("listt.txt"):
                    continue
                name = name.replace(".jpg", ".ppm")
                smiling_files.append(name)

        # TODO: Read the non-smiling faces file and extract filenames
        # Hint: Same process as above - open file, read lines, process filenames
        non_smiling_files = [] # type: list
        with open(non_smiling_labels_file, "r") as f:
            for line in f:
                name = line.strip()
                if not name or name.endswith("list.txt"):
                    continue
                name = name.replace(".jpg", ".ppm")
                non_smiling_files.append(name)

        # Create image paths and labels
        self.image_paths = []
        self.labels = []

        # TODO: Add smiling faces to the dataset with label = 1
        # Hint: Loop through smiling_files, create full file paths using os.path.join
        # Check if file exists with os.path.isfile, then append to image_paths and labels

        for file_path in smiling_files:
            full_path = os.path.join(self.faces_folder, file_path)
            if os.path.isfile(full_path):
                self.image_paths.append(full_path)
                self.labels.append(1)

        # TODO: Add non-smiling faces to the dataset with label = 0
        # Hint: Same process as above but with label = 0

        for file_path in non_smiling_files:
            full_path = os.path.join(self.faces_folder, file_path)
            if os.path.isfile(full_path):
                self.image_paths.append(full_path)
                self.labels.append(0)

    def __len__(self):
        # TODO: Return the total number of images in the dataset
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO: Get the image path and label for the given index
        # Hint: Use idx to index into your image_paths and labels lists
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # TODO: Load and convert the image to RGB
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
