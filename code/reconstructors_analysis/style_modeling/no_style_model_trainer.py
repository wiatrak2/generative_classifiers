from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np
import datetime

from distsup.reconstructors_analysis import style_modeling


class PixelCnnNoStyleModelTrainer:
    def __init__(
        self,
        model,
        data_loader,
        test_data_loader=None,
        num_levels=16,
        loss_reduction="sum",
        monitor=None,
        serialization_filename=None,
        serialization_path=None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.test_data_loader = test_data_loader
        self.dataset: style_modeling.qmnist.dataset.QMNISTSingleAuthorSet = data_loader.dataset
        self.num_levels = num_levels
        self.loss_reduction = loss_reduction

        self.recon_losses = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.serialization_filename = serialization_filename
        self.serialzation_path = serialization_path

        self.monitor = monitor

    def train_mode(self):
        self.model.train()

    def eval_mode(self):
        self.model.eval()

    def train(self, optimizers, epochs):
        try:
            self.train_mode()
            for epoch in range(1, epochs + 1):
                for batch_num, data in enumerate(self.data_loader):
                    inputs, targets = self.get_inputs_and_targets(data)
                    for optimizer in optimizers:
                        optimizer.zero_grad()
                    cond = self.get_conditioning(data)
                    model_output = self.model(inputs, cond)
                    loss = self.get_loss(model_output, targets)
                    loss.backward()
                    loss_item = loss.item()
                    for optimizer in optimizers:
                        optimizer.step()

                    if self.monitor is not None:
                        self.monitor.monitor("loss", loss_item)
                        self.monitor.monitor("batch_num", batch_num)
                    if batch_num % 100 == 0:
                        print(
                            f"Train Epoch: {epoch} [{batch_num * self.data_loader.batch_size}/{len(self.data_loader.dataset)} "
                            f"({100.0 * batch_num / len(self.data_loader):.0f}%)]\tLoss: {loss_item:.6f}"
                        )
                self.recon_losses.append(self.test_model_reconstruction())
        except KeyboardInterrupt:
            print("Interrupting")

    def get_conditioning(self, data):
        target_class_label = data["target_image_label"]
        batch_size = target_class_label.size(0)
        target_class_label_one_hot = (
            F.one_hot(target_class_label, 10)
            .view(batch_size, 1, 1, -1)
            .float()
            .to(self.device)
        )
        return (target_class_label_one_hot,)

    def get_inputs_and_targets(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        if data["target_image"].size(-1) == 1 and data["target_image"].size(1) != 1:
            data["target_image"] = data["target_image"].permute(0, 3, 1, 2)
        target_img = data["target_image"].to(self.device)
        return self.model.get_inputs_and_targets(target_img.permute(0, 2, 3, 1))

    def get_loss(self, model_output, targets: torch.Tensor):
        loss = self.model.loss(model_output, targets)
        if self.loss_reduction == "sum":
            return loss.sum()
        elif self.loss_reduction == "mean":
            return loss.mean()
        else:
            raise "Wrong loss reduction"

    def sample(self, data, class_label=None):
        if class_label is None:
            class_label = np.random.randint(0, 10)
        class_label = torch.tensor([[class_label]])
        self.eval_mode()
        sample = self.get_zeros_like_single_input_shape()
        cond_data = {"target_image_label": class_label}
        cond = self.get_conditioning(cond_data)
        for i in range(28):
            for j in range(28):
                pbb = F.softmax(
                    self.model(sample, cond)[:, i, j, :, :].squeeze(), dim=0
                )
                pixel = torch.multinomial(pbb, 1)
                sample[:, i, j, :] = pixel / float(self.num_levels - 1)
        return sample

    def get_zeros_like_single_input_shape(self):
        return torch.zeros_like(
            x=self.data_loader.dataset[0]["target_image"][None, :].permute(0, 2, 3, 1)
        ).to(device)

    def compute_image_pbb(self, sample, conds):
        with torch.no_grad():
            self.eval_mode()
            # self.image is normalized to 0-1 range, we need integer value of pixels
            _, img_pixels_int = self.get_inputs_and_targets({"target_image": sample})
            model_out = tuple(self.model(sample, conds))[
                0
            ]  # torch.Size([1, 28, 28, 1, 16])
            probabilities = F.log_softmax(model_out.squeeze(), dim=-1).reshape(
                -1, model_out.size(-1)
            )
            # Reshape to 1D vector
            img_pixels_int_vec = img_pixels_int.reshape(-1)
            image_pixels_pbb = probabilities[
                torch.arange(probabilities.size(0)), img_pixels_int_vec
            ]
            return image_pixels_pbb

    def get_sample_log_pbb(self, data):
        with torch.no_grad():
            self.eval_mode()
            conds = self.get_conditioning(data)
            return torch.sum(self.compute_image_log_pbb(data["target_image"], conds))

    def test_generated_sample_class_pbb(self, data, class_label=None):
        with torch.no_grad():
            self.eval_mode()
            if class_label is None:
                class_label = np.random.randint(0, 10)
            class_label = torch.tensor([[class_label]])
            generated_sample = self.sample(data, class_label)
            label_pbb = {}
            for i in range(10):
                cond_data = {
                    "target_image_label": torch.tensor([[i]]),
                }
                conds = self.get_conditioning(cond_data)
                label_pbb[i] = torch.sum(
                    self.compute_image_pbb(generated_sample, conds)
                )
            return generated_sample, label_pbb

    def get_reconstruction_loss(self, model_output, targets: torch.Tensor):
        return self.model.loss(model_output, targets)

    def nats_per_pix(self, data):
        inputs, targets = self.get_inputs_and_targets(data)
        cond = self.get_conditioning(data)
        model_reconstrution = self.model(inputs, cond)
        rec_loss = self.get_reconstruction_loss(model_reconstrution, targets)
        return torch.mean(rec_loss.view(rec_loss.size(0), -1), dim=-1)  # nats/pix

    def test_model_reconstruction(self, min_num_samples=100):
        with torch.no_grad():
            assert self.test_data_loader is not None
            reconstruction_loss = torch.tensor([]).to(self.device)
            for data in self.test_data_loader:
                inputs, targets = self.get_inputs_and_targets(data)
                cond = self.get_conditioning(data)
                model_reconstrution = self.model(inputs, cond)
                recon_batch = self.get_reconstruction_loss(model_reconstrution, targets)
                reconstruction_loss = torch.cat(
                    (
                        reconstruction_loss,
                        torch.mean(recon_batch.view(recon_batch.size(0), -1), dim=-1),
                    )
                )  # nats/pix
                if (
                    min_num_samples > 0
                    and reconstruction_loss.size(0) >= min_num_samples
                ):
                    break
            return reconstruction_loss

    def _sample_to_input_shape(self, sample):
        return sample[None, :].permute(0, 2, 3, 1).to(self.device)

    def test_model_classification(self, num_samples=100):
        """
        Computes probability of test images X_0, X_1, ... for all possible digit labels d=0...9.
        Each time P(X_i | d) is computed. The model prediction is value of d that maximizes the term.
        Ths approach bases on the Bayes theorem.
        """
        with torch.no_grad():
            self.eval_mode()
            test_data = []
            for i, data in enumerate(self.test_data_loader.dataset):
                if i >= num_samples:
                    break
                label_log_pbb = torch.empty(10)
                sample = self._sample_to_input_shape(data["target_image"])
                for label in range(10):
                    cond_data = {
                        "target_image_label": torch.tensor([[label]]),
                    }
                    conds = self.get_conditioning(cond_data)
                    label_log_pbb[label] = torch.sum(
                        self.compute_image_pbb(sample, conds)
                    )

                prediction = torch.argmax(label_log_pbb)
                test_data.append(
                    {
                        "data": data,
                        "log_pbbs": label_log_pbb,
                        "prediction": prediction.item(),
                        "correct": (prediction == data["target_image_label"]).item(),
                    }
                )
        return test_data

    def _get_serialization_dict(self, optimizers=None):
        serialize_dict = {
            "model_state_dict": self.model.state_dict(),
            "recon_losses": self.recon_losses,
        }
        if optimizers is not None:
            for i, optimizer in enumerate(optimizers):
                serialize_dict[f"optimizer_{i}"] = optimizer.state_dict()
        return serialize_dict

    def serialize(self, optimizers=None):
        serialize_dict = self._get_serialization_dict(optimizers)

        serialization_dirctory = self.serialzation_path or "models/"
        serialization_filename = self.serialization_filename or "single_author_digits"
        serialization_filename += type(self.model).__name__
        serialization_path = (
            serialization_dirctory
            + serialization_filename
            + datetime.datetime.now().strftime("_%Y-%m-%d_%H:%M:%S")
            + ".pkl"
        )
        try:
            torch.save(serialize_dict, serialization_path)
        except Exception as e:
            print(f"Could not serialize model. Exception raised: {e}")
            return
        print(f"Model serialized as {serialization_path}")


class PixelCnnNoConditioningModelTrainer(PixelCnnNoStyleModelTrainer):
    def get_conditioning(self, data):
        return ()

    def sample(self, data, *args, **kwargs):
        self.model.eval()
        sample = self.get_zeros_like_single_input_shape()
        cond = self.get_conditioning(data)
        for i in range(28):
            for j in range(28):
                pbb = F.softmax(
                    self.model(sample, cond)[:, i, j, :, :].squeeze(), dim=0
                )
                pixel = torch.multinomial(pbb, 1)
                sample[:, i, j, :] = pixel / float(self.num_levels - 1)
        return sample

    def compute_image_pbb(self, sample, conds=None):
        conds = ()
        return super().compute_image_pbb(sample, conds)

    def test_model_classification(self, num_samples=100):
        raise Exception(
            "The reconstructor is not conditioned - unable to compute probabilities."
        )


class PixelCnnNoStyleModelWithEmbeddingTrainer(PixelCnnNoStyleModelTrainer):
    def __init__(self, *args, embedding: nn.Embedding, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding = embedding
        self.test_data_loader = self.data_loader

    def train_mode(self):
        super().train_mode()
        self.embedding.train()

    def eval_mode(self):
        super().eval_mode()
        self.embedding.eval()

    def get_conditioning(self, data):
        batch_size = data["target_image_global_id"].size(0)
        digit_cond = self.embedding(
            data["target_image_global_id"].to(self.device)
        ).view(batch_size, 1, 1, -1)
        return (digit_cond,)

    def _get_serialization_dict(self, optimizers=None):
        serialize_dict = super()._get_serialization_dict(optimizers)
        serialize_dict["embedding"] = self.embedding.state_dict()
        return serialize_dict
