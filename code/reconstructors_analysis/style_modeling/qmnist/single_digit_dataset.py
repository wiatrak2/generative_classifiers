#!/usr/bin/env python

import os
import re
import zipfile
import copy
import collections
import logging
from collections import defaultdict

import torchvision
import torch.utils.data
import PIL.Image
import numpy as np

from distsup.alphabet import Alphabet
from distsup.configuration import Globals
from distsup import utils


class MNISTSingleDigitImageSetDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root="/content/drive/My Drive/master_thesis/data/qmnist_set_single_digit.rand0.25000.1000.zip",
        split="train",
        set_size=10,
        sample_size=(28, 28),
        max_samples=None,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()]
        ),
    ):
        self.root = root
        self.file = zipfile.ZipFile(root)
        self.transform = transform
        self.set_size = set_size
        self.sample_size = sample_size
        self.alphabet = Alphabet(
            input_dict={
                "*": 0,
                "1": 1,
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 7,
                "8": 8,
                "9": 9,
                "0": 10,
            },
            blank="*",
            space=(),
        )

        fnm_pattern = r".*\/sample_(?P<sample_num>\d+)_writer(?P<writer_id>\d+)_char(?P<char_id>\d)_img(?P<img_num>\d+)?\.jpg"
        fnm_matcher = re.compile(fnm_pattern)

        fnm_glob = re.compile(f".*/{split}/.*\.jpg")
        filenames = sorted(
            [f.filename for f in self.file.filelist if fnm_glob.match(f.filename)]
        )
        self.data = defaultdict(list)
        for filename in filenames[:max_samples]:
            metadata = {
                "img_filename": filename,
                "dataset_idx_filename": filename.split("_img")[0] + ".dataset_idx",
            }

            match = re.match(fnm_matcher, filename)
            if not match:
                raise ValueError(
                    f"Filename {filename} did not match the expected pattern."
                )

            metadata.update(match.groupdict())
            sample_num = metadata["sample_num"]
            if sample_num is None:
                continue
            self.data[sample_num].append(metadata)

        self.file.close()
        self.file = None

        self.metadata = {
            "alignment": {
                "type": "categorical",
                "num_categories": len(self.alphabet),
            }  # 10 digits plus 1 blank
        }

    def __len__(self):
        return len(self.data)

    def sample(self, idx):
        return list(self.data.values())[idx]

    def __getitem__(self, idx):
        if not self.file:  # Reopen to work with multiprocessing
            self.file = zipfile.ZipFile(self.root)

        item_data = self.sample(idx)

        item = {}

        target_image_filename = item_data[0]["img_filename"]
        with self.file.open(target_image_filename) as f:
            target_image = PIL.Image.open(f).convert("RGB")
        if self.transform:
            target_image = self.transform(target_image)
        item["target_image"] = target_image

        stacked_images = torch.empty((self.set_size - 1, *self.sample_size))
        for i, sample in enumerate(item_data[1:]):
            with self.file.open(sample["img_filename"]) as f:
                image = PIL.Image.open(f).convert("RGB")
            if self.transform:
                image = self.transform(image)
            stacked_images[i, :] = image
        item["images"] = stacked_images

        dataset_idx_filename = item_data[0].get("dataset_idx_filename")
        if dataset_idx_filename:
            with self.file.open(dataset_idx_filename) as f:
                dataset_idx = np.loadtxt(f, dtype=np.int)
            target_idx, images_idx = dataset_idx[0], dataset_idx[1:]
            item["target_idx"] = dataset_idx
            item["images_idx"] = images_idx

        item["label"] = torch.tensor(int(item_data[0].get("char_id")))
        item["writer_id"] = torch.tensor(int(item_data[0].get("writer_id")))

        return item
