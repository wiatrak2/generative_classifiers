from copy import deepcopy
from collections import Counter

import torch
import torchvision
import torch.utils.data
import numpy as np

from notebooks.qmnist import QMNIST


def get_dataset(sample_size=10, test_split=True, all_samples_for_test_set=False):
    qmnist = QMNIST("_qmnist", what="nist", compat=False, download=True)
    qmnist_author_dataset = QMNISTSingleAuthorSet(qmnist, sample_size=sample_size)
    if not test_split:
        return qmnist_author_dataset
    all_authors_range = qmnist_author_dataset.authors_range
    split_point = int(len(all_authors_range) * 0.8)
    train_authors = all_authors_range[:split_point]
    test_authors = all_authors_range[split_point:]
    qmnist_author_dataset.authors_range = train_authors
    qmnist_author_dataset_test = QMNISTSingleAuthorSet(
        qmnist,
        sample_size=sample_size,
        authors_range=test_authors,
        all_samples_to_style=all_samples_for_test_set,
    )
    return qmnist_author_dataset, qmnist_author_dataset_test


class QMNISTSingleAuthorSet(torch.utils.data.Dataset):
    COLUMNS = [
        "Character class",
        "NIST HSF series",
        "NIST writer ID",
        "Digit index for this writer",
        "NIST class code",
        "Global NIST digit index",
        "Duplicate",
        "Unused",
    ]
    AUTHOR_ID_COLUMN = 2
    PER_AUTHOR_SAMPLES = 10

    def __init__(
        self,
        dataset=None,
        sample_size=10,
        sample_image_size=(28, 28),
        authors_range=None,
        all_samples_to_style=False,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()]
        ),
    ):
        if dataset is None:
            dataset = QMNIST("_qmnist", what="nist", compat=False, download=True)
        self.dataset = dataset
        self.sample_size = sample_size
        self.sample_image_size = sample_image_size
        self.all_samples_to_style = all_samples_to_style
        self.transform = transform

        if authors_range is None:
            self.authors_range = []
            author_first_sample = 0
            for i in range(len(self.dataset) - 1):
                assert (
                    self.dataset[i][1][self.AUTHOR_ID_COLUMN]
                    <= self.dataset[i + 1][1][self.AUTHOR_ID_COLUMN]
                )
                if (
                    self.dataset[i][1][self.AUTHOR_ID_COLUMN]
                    < self.dataset[i + 1][1][self.AUTHOR_ID_COLUMN]
                ):
                    self.authors_range.append((author_first_sample, i + 1))
                    author_first_sample = i + 1
            self.authors_range.append((author_first_sample, len(self.dataset)))
            self._filter_authors()
        else:
            self.authors_range = authors_range

    def __len__(self):
        return len(self.authors_range) * self.PER_AUTHOR_SAMPLES

    def _filter_authors(self):
        self.authors_range = [
            (begin, end)
            for (begin, end) in self.authors_range
            if end - begin >= self.sample_size
        ]

    def __getitem__(self, idx):
        author_num = idx // self.PER_AUTHOR_SAMPLES
        sample_images_idx = np.random.permutation(
            np.arange(*self.authors_range[author_num])
        )[: self.sample_size]

        target_sample_idx = sample_images_idx[0]
        target_image, target_labels = self.dataset[target_sample_idx]
        stacked_images = torch.empty((self.sample_size - 1, *self.sample_image_size))
        labels = torch.zeros(self.sample_size - 1)
        global_ids = torch.zeros(self.sample_size - 1)

        if self.all_samples_to_style:
            sample_images_idx = [target_sample_idx] + [
                sample_idx
                for sample_idx in range(*self.authors_range[author_num])
                if sample_idx != target_sample_idx
            ]
            stacked_images = torch.empty(
                (len(sample_images_idx) - 1, *self.sample_image_size)
            )
            labels = torch.zeros(len(sample_images_idx) - 1)
            global_ids = torch.zeros(len(sample_images_idx) - 1)

        label_idx = self.COLUMNS.index("Character class")
        author_idx = self.COLUMNS.index("NIST writer ID")
        global_id_idx = self.COLUMNS.index("Global NIST digit index")

        author_id = target_labels[author_idx]
        for i, sample_idx in enumerate(sample_images_idx[1:]):
            sample_img, sample_lables = self.dataset[sample_idx]
            stacked_images[i, :] = self.transform(sample_img)
            labels[i] = sample_lables[label_idx]
            global_ids[i] = sample_lables[global_id_idx]
            assert sample_lables[author_idx] == author_id
        return {
            "images": stacked_images,
            "labels": labels,
            "global_ids": global_ids,
            "target_image": self.transform(target_image),
            "target_image_label": target_labels[label_idx],
            "target_image_global_id": target_labels[global_id_idx],
            "author_id": author_id,
        }


def get_full_dataset(
    sample_size=10,
    test_split=True,
    all_samples_for_test_set=False,
    max_train_samples=None,
    max_test_samples=None,
):
    qmnist = QMNIST("_qmnist", what="nist", compat=False, download=True)
    qmnist_author_dataset = QMNISTSingleAuthorFullSet(qmnist, sample_size=sample_size)
    if not test_split:
        return qmnist_author_dataset
    split_point = int(len(qmnist_author_dataset) * 0.8)
    author_id_range = qmnist_author_dataset.author_id_to_range
    train_dataset = QMNISTSingleAuthorFullSet(
        qmnist,
        sample_size=sample_size,
        author_id_to_range=author_id_range,
        samples_range=(0, split_point),
        max_samples=max_train_samples,
    )
    test_dataset = QMNISTSingleAuthorFullSet(
        qmnist,
        sample_size=sample_size,
        author_id_to_range=author_id_range,
        samples_range=(split_point, len(qmnist_author_dataset.dataset)),
        all_samples_to_style=all_samples_for_test_set,
        max_samples=max_test_samples,
    )
    return train_dataset, test_dataset


class QMNISTSingleAuthorFullSet(QMNISTSingleAuthorSet):
    def __init__(
        self,
        *args,
        author_id_to_range=None,
        samples_range=None,
        max_samples=None,
        **kwargs
    ):
        self.author_id_to_range = author_id_to_range
        super().__init__(*args, **kwargs)
        self.samples_range = samples_range or (0, len(self.dataset))
        if max_samples is not None:
            range_start, range_end = self.samples_range
            range_end = min(range_end, range_start + max_samples)
            self.samples_range = (range_start, range_end)

    def __len__(self):
        range_start, range_end = self.samples_range
        return range_end - range_start

    def _filter_authors(self):
        max_author_id = self.dataset[-1][1][self.AUTHOR_ID_COLUMN]
        self.author_id_to_range = np.zeros(max_author_id + 2, dtype=int)
        for (begin, end) in self.authors_range:
            author_id = self.dataset[begin][1][self.AUTHOR_ID_COLUMN]
            self.author_id_to_range[author_id] = begin
            self.author_id_to_range[author_id + 1] = end
        self.author_id_to_range[-1] = len(self.dataset)

    def __getitem__(self, idx):
        range_start, _ = self.samples_range
        idx += range_start

        target_image, target_labels = self.dataset[idx]
        author_id = target_labels[self.AUTHOR_ID_COLUMN]
        author_range_start, author_range_end = (
            self.author_id_to_range[author_id],
            self.author_id_to_range[author_id + 1],
        )

        sample_images_idx = np.random.permutation(
            np.arange(author_range_start, author_range_end)
        )[: self.sample_size - 1]

        stacked_images = torch.empty((self.sample_size - 1, *self.sample_image_size))
        labels = torch.zeros(self.sample_size - 1)
        global_ids = torch.zeros(self.sample_size - 1)

        if self.all_samples_to_style:
            sample_images_idx = [
                sample_idx
                for sample_idx in range(author_range_start, author_range_end)
                if sample_idx != idx
            ]
            stacked_images = torch.empty(
                (len(sample_images_idx), *self.sample_image_size)
            )
            labels = torch.zeros(len(sample_images_idx))
            global_ids = torch.zeros(len(sample_images_idx))

        label_idx = self.COLUMNS.index("Character class")
        author_idx = self.COLUMNS.index("NIST writer ID")
        global_id_idx = self.COLUMNS.index("Global NIST digit index")

        for i in range(self.sample_size - 1):
            sample_idx = sample_images_idx[i % len(sample_images_idx)]
            sample_img, sample_lables = self.dataset[sample_idx]
            stacked_images[i, :] = self.transform(sample_img)
            labels[i] = sample_lables[label_idx]
            global_ids[i] = sample_lables[global_id_idx]
            assert sample_lables[author_idx] == author_id
        return {
            "images": stacked_images,
            "labels": labels,
            "global_ids": global_ids,
            "target_image": self.transform(target_image),
            "target_image_label": target_labels[label_idx],
            "target_image_global_id": target_labels[global_id_idx],
            "author_id": author_id,
        }


def unsqueeze_sample(sample):
    def _unsqueeze_field(field, expected_len):
        if len(sample[field].shape) < expected_len:
            sample[field] = sample[field].unsqueeze(0)

    _unsqueeze_field("images", 4)
    _unsqueeze_field("target_image", 4)
    _unsqueeze_field("target_image_label", 1)
    _unsqueeze_field("target_image_global_id", 1)


def get_samples_with_different_style_input(
    dataset, num_samples, index=None, start_with_target_sample=False
):
    if index is None:
        index = np.random.randint(0, len(dataset))
    origin_sample = dataset[index]
    unsqueeze_sample(origin_sample)
    if start_with_target_sample:
        yield {
            "images": origin_sample["target_image"],
            "target_image": origin_sample["target_image"].permute(0, 2, 3, 1),
            "target_image_label": origin_sample["target_image_label"],
        }
    origin_sample["target_image"] = origin_sample["target_image"].permute(0, 2, 3, 1)
    for _ in range(num_samples - (start_with_target_sample)):
        new_sample = dataset[index]
        new_sample["target_image"] = origin_sample["target_image"]
        new_sample["target_image_label"] = origin_sample["target_image_label"]
        new_sample["target_image_global_id"] = origin_sample["target_image_global_id"]
        yield new_sample


def get_all_digits_dataset(test_split=True):
    qmnist = QMNIST("_qmnist", what="nist", compat=False, download=True)
    qmnist_author_dataset = QMNISTSingleAuthorAllDigitsSet(qmnist)
    if not test_split:
        return qmnist_author_dataset
    all_authors_range = qmnist_author_dataset.authors_range
    split_point = int(len(all_authors_range) * 0.8)
    train_authors = all_authors_range[:split_point]
    test_authors = all_authors_range[split_point:]
    qmnist_author_dataset.authors_range = train_authors
    qmnist_author_dataset_test = QMNISTSingleAuthorAllDigitsSet(
        qmnist, authors_range=test_authors
    )
    return qmnist_author_dataset, qmnist_author_dataset_test


class QMNISTSingleAuthorAllDigitsSet(QMNISTSingleAuthorSet):
    def __init__(
        self,
        dataset=None,
        sample_image_size=(28, 28),
        authors_range=None,
        transform=torchvision.transforms.Compose(
            [torchvision.transforms.Grayscale(), torchvision.transforms.ToTensor()]
        ),
    ):
        super().__init__(
            dataset=dataset,
            sample_size=11,
            sample_image_size=sample_image_size,
            authors_range=authors_range,
            transform=transform,
        )

    def _filter_authors(self):
        filtered_authors_range = []
        label_idx = self.COLUMNS.index("Character class")
        for author_range_start, author_range_end in self.authors_range:
            author_digits_counter = Counter()
            author_digits_idx = {digit: [] for digit in range(10)}
            for idx in range(author_range_start, author_range_end):
                _, labels = self.dataset[idx]
                author_digits_counter[labels[label_idx].item()] += 1
                author_digits_idx[labels[label_idx].item()].append(idx)
            if (
                len(author_digits_counter.values()) == 10
                and max(author_digits_counter.values()) >= 2
            ):
                author_style_samples_idx = [
                    np.random.choice(author_digits_idx[digit]) for digit in range(10)
                ]
                filtered_authors_range.append(
                    (author_range_start, author_range_end, author_style_samples_idx)
                )
        self.authors_range = filtered_authors_range

    def __getitem__(self, idx):
        author_num = idx // self.PER_AUTHOR_SAMPLES
        author_range_start, author_range_end, author_style_samples = self.authors_range[
            author_num
        ]

        target_image_idx = np.random.choice(
            [
                idx
                for idx in range(author_range_start, author_range_end)
                if idx not in author_style_samples
            ]
        )

        stacked_images = torch.empty((10, *self.sample_image_size))
        labels = torch.zeros(10)
        global_ids = torch.zeros(10)
        label_idx = self.COLUMNS.index("Character class")
        author_idx = self.COLUMNS.index("NIST writer ID")
        global_id_idx = self.COLUMNS.index("Global NIST digit index")

        target_image, target_labels = self.dataset[target_image_idx]
        author_id = target_labels[author_idx]

        for digit in range(10):
            sample_idx = author_style_samples[digit]
            sample_img, sample_lables = self.dataset[sample_idx]
            stacked_images[digit, :] = self.transform(sample_img)
            labels[digit] = sample_lables[label_idx]
            global_ids[digit] = sample_lables[global_id_idx]
            assert sample_lables[author_idx] == author_id

        return {
            "images": stacked_images,
            "labels": labels,
            "global_ids": global_ids,
            "target_image": self.transform(target_image),
            "target_image_label": target_labels[label_idx],
            "target_image_global_id": target_labels[global_id_idx],
            "author_id": author_id,
        }
