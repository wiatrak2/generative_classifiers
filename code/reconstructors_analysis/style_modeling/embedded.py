import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup.reconstructors_analysis import style_modeling


class GumbelSoftmax(nn.Softmax):
    def __init__(self, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, input):
        return F.gumbel_softmax(input, self.temperature, dim=self.dim)


class MaskedEmbedding(nn.Embedding):
    def __init__(self, *args, mask=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def forward(self, *args, **kwargs):
        embedding = super().forward(*args, **kwargs)
        if self.mask is not None:
            embedding = self.mask(embedding)
        return embedding


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        embedding: MaskedEmbedding,
        dataset: style_modeling.qmnist.dataset.QMNISTSingleAuthorSet,
        samples_range=None,
    ):
        self.embedding = embedding.weight.detach().clone().cpu()
        self.mask = embedding.mask
        self.dataset = dataset
        self.samples_range = samples_range or (0, embedding.num_embeddings)

    def __len__(self):
        range_start, range_end = self.samples_range
        return range_end - range_start

    def __getitem__(self, idx):
        range_start, _ = self.samples_range
        idx += range_start
        label = self.dataset[idx]["target_image_label"]
        embedding = self.embedding[idx]
        if self.mask is not None:
            embedding = self.mask(embedding)
        return embedding, label
