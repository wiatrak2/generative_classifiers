import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from distsup import utils


class OneHotReconstructor(nn.Module):
    def __init__(
        self,
        embedding_dim,
        image_height=28,
        len_reduction=4,
        reconstructor={
            "class_name": "distsup.modules.reconstructors.ColumnGatedPixelCNN",
            "quantizer": {
                "class_name": "distsup.modules.quantizers.SoftmaxUniformQuantizer",
                "num_levels": 16,
            },
        },
        device="cpu",
        ignore_alignment=False,
        count_blanks_alignment=True,
        advantage_digits=True,
        alignment_noise_pbb=None,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.image_height = image_height
        self.len_reduction = len_reduction
        cond_channels_spec = [
            {"cond_dim": embedding_dim, "reduction_factor": len_reduction}
        ]
        rec_params = {"image_height": image_height, "cond_channels": cond_channels_spec}

        self.device = device
        self.reconstructor = utils.construct_from_kwargs(
            reconstructor, additional_parameters=rec_params
        ).to(device)

        self.ignore_alignment = ignore_alignment
        self.count_blanks_alignment = count_blanks_alignment
        self.advantage_digits = advantage_digits
        self.alignment_noise_pbb = alignment_noise_pbb
        self.mask = torch.ones(embedding_dim).long().to(device)
        if not self.count_blanks_alignment:
            self.mask[0] = 0

    def pad_batch(self, image_batch):
        padded_size = (
            (image_batch.size(1) + self.len_reduction - 1) // self.len_reduction
        ) * self.len_reduction
        image_batch = F.pad(
            image_batch, [0, 0, 0, 0, 0, padded_size - image_batch.size(1)]
        )
        assert image_batch.size(1) % self.len_reduction == 0
        return image_batch

    def transform_alignment(self, alignment, count_blanks=True):
        """
    Transforms the alignment description of the batch.
    Alignments are passed as padded tensor of size BS x AlignmentLen
    Each of BS padded alignments is a tensor with label of digit/letter in a column of pixels in corresponding image.
    This tensor (of lenght AlignmentLen) is transformed to a AlignmentLen x AlphabetSize matrix,
    where each label of digit/letter is transformed to one-hot vector (0 -> [1,0,0,0..], 1 -> [0,1,0,0..]).
    Then the matrix is reshaped to AlignmentLen x 1 x AlphabetSize tensor and reduced.
    The reduction has stride = self.len_reduction and is transforming len_rediction one-hot vectors to a vector
    representing the digit(s) placed in corresponding len_reduction pixel columns of the image.
    In used alphabets datasets the label `0` is assigned to "blank", but if `self.count_blanks_alignment` is False,
    the reduction transforms the following collumns of empty space to a vector of zeros ([1,0,0,0..] -> [0,0,0,0..]).
    If a digit occures within a len_reduction wide window, the reduction is assigning its one-hot vector to the transformed alignment,
    if `self.count_blanks_alignment` is True and some balnk columns also occures, the common label will be a combination of
    digit's and blank space one-hot vectors. if two digits occur within the window (unlikly), the reduction would make
    a logical or of their one-hot representation regardless of the `self.count_blanks_alignment` value
    (e.g. if there are digits '1' and '2' within the window, the reduction would produce [0,1,0,0,0..], [0,0,1,0,0..] -> [0,1,1,0,0..])
    """
        one_hot_alignment = F.one_hot(alignment, self.embedding_dim)
        bs, al_len, num_classes = one_hot_alignment.size()
        padded_alignment = self.pad_batch(
            one_hot_alignment.reshape(bs, al_len, 1, num_classes)
        )
        reduced_al_len = padded_alignment.size(1) // self.len_reduction
        reduced_alignment = torch.empty((bs, reduced_al_len, 1, num_classes)).to(
            self.device
        )
        assert reduced_alignment.size(1) * self.len_reduction == padded_alignment.size(
            1
        )
        for i in range(0, al_len, self.len_reduction):
            reduced_idx = i // self.len_reduction
            reduced_alignment[:, reduced_idx, :, :] = (
                torch.max(
                    padded_alignment[:, i : i + self.len_reduction, :, :], dim=1
                ).values
                * self.mask
            )
            if not self.advantage_digits:
                mode_alignment = torch.mode(
                    padded_alignment[:, i : i + self.len_reduction, :, :], dim=1
                ).values
                if torch.sum(mode_alignment) > 0:
                    reduced_alignment[:, reduced_idx, :, :] = mode_alignment
            if self.alignment_noise_pbb is not None:
                noise_idx = np.random.randint(0, self.embedding_dim)
                if np.random.rand() < self.alignment_noise_pbb:
                    reduced_alignment[:, reduced_idx, :, noise_idx] = 1
        if self.ignore_alignment:
            return torch.zeros_like(reduced_alignment).to(self.device)
        return reduced_alignment

    def get_inputs_and_targets(self, x, **kwargs):
        return self.reconstructor.get_inputs_and_targets(x, **kwargs)

    def reconstruction_loss(self, batch):
        feats = batch["features"].to(self.device)
        labels = batch["alignment"].long().to(self.device)
        feats = self.pad_batch(feats)
        inputs, targets = self.get_inputs_and_targets(feats)
        conds = self.transform_alignment(labels)
        batch_rec = self.reconstructor(inputs, (conds,))
        return self.reconstructor.loss(batch_rec, targets)

    def sample(self, labels, cut_blank=False, start_cond=None):
        self.reconstructor.eval()
        if cut_blank:
            digit_labels = torch.where(labels != 0)[0]
            digit_start, digit_end = digit_labels[0], digit_labels[-1]
            labels = labels[
                (digit_start - 5 * self.len_reduction) : (
                    digit_end + 5 * self.len_reduction
                )
            ]
        labels = labels.to(self.device)
        sample_cond = self.transform_alignment(labels[None, :].long())
        cond_len = len(labels)
        sample_img = torch.zeros((1, cond_len, self.image_height, 1)).to(self.device)
        sample_img = self.pad_batch(sample_img)
        if start_cond is not None:
            start_img, start_step = start_cond
            start_img = self.pad_batch(start_img)
            return self.reconstructor.sample(
                start_img, (sample_cond,), start_step=start_step
            )
        return self.reconstructor.sample(sample_img, (sample_cond,))

    def forward(self, feats, labels=None):
        feats = self.pad_batch(feats)
        inputs, _ = self.get_inputs_and_targets(feats)
        conds = (self.transform_alignment(labels),)
        if labels is None:
            conds = ()
        return self.reconstructor(inputs, conds)
