class ReconstructorTrainer:
    def __init__(
        self,
        model,
        data_loader,
        serialization_path=None,
        serialization_filename=None,
        serialize_epochs=None,
        serialize_sample=None,
    ):
        self.model = model
        self.data_loader = data_loader
        self.serialization_path = (
            serialization_path or "/content/drive/My Drive/master_thesis/models/"
        )
        self.serialization_filename = serialization_filename
        self.serialize_epochs = serialize_epochs
        self.serialize_sample = serialize_sample
        self.losses = []

    def train(self, optimizer, epochs, scheduler=None):
        self.model.train()
        try:
            for epoch in range(1, epochs + 1):
                for batch_idx, batch in enumerate(self.data_loader):

                    optimizer.zero_grad()
                    loss = self.model.reconstruction_loss(batch)
                    loss_value = loss.sum().item()
                    self.losses.append(loss_value)
                    loss.sum().backward()
                    optimizer.step()
                    ml_monitor.monitor("loss", loss_value),
                    ml_monitor.monitor("batch_num", batch_idx)
                    if batch_idx % 100 == 0:
                        print(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                epoch,
                                batch_idx * self.data_loader.batch_size,
                                len(self.data_loader.dataset),
                                100.0 * batch_idx / len(self.data_loader),
                                loss_value,
                            )
                        )

                if scheduler:
                    print("Applying LR scheduler...")
                    scheduler.step()

                if self.serialize_epochs and epoch % self.serialize_epochs == 0:
                    print("Serializing model...")
                    self.serialize(epoch, optimizer)

        except KeyboardInterrupt:
            print("Interrupting...")
        return self.losses

    def serialize(self, epoch, optimizer):
        serialize_dict = {
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "losses": self.losses,
        }
        if not self.serialization_filename:
            self.serialization_filename = type(self.model.reconstructor).__name__
        serialization_path = (
            self.serialization_path
            + self.serialization_filename
            + datetime.datetime.now().strftime("_%Y-%m-%d_%H:%M:%S")
            + ".pkl"
        )
        try:
            torch.save(serialize_dict, serialization_path)
        except Exception as e:
            print(f"Could not serialize model. Exception raised: {e}")
            return
        print(f"Model serialized as {serialization_path}")
        if self.serialize_sample is not None:
            generated_sample = self.model.sample(self.serialize_sample)
            sample_path = serialization_path.replace(".pkl", ".png")
            plt.imsave(
                sample_path,
                generated_sample.squeeze().data.cpu().transpose(0, 1),
                cmap="gray",
            )
            print(f"Sample image serialized as {sample_path}")
