class ImagePbbCalculator:
    def __init__(self, image, labels, alignment_rle, ignored_symbols=[]):
        if len(image.shape) == 3:
            # image should be 1 x W x H x 1
            image = image.unsqueeze(0)
        if len(labels.shape) == 1:
            # labels should be 1 x W
            labels = labels.unsqueeze(0)

        self.image = image.to(device)
        self.labels = labels.long().to(device)

        self.image_width = image.size(1)

        self.zero_img = torch.zeros_like(image).to(device)
        # get tuples (digit, start_pos, end_pos), where start_pos/end_pos are corresponding labels range
        self.digits_positions = [
            (labels[:, start.long()], start, end)
            for start, end in alignment_rle
            if start != end
        ]

        self.ignored_symbols = ignored_symbols

    def is_position_ignored(self, position):
        return self.digits_positions[position][0] in self.ignored_symbols

    def compute_image_pbb(self, model, labels, pixels_range=None):
        with torch.no_grad():
            # self.image is normalized to 0-1 range, we need integer value of pixels
            img_pixels_int = model.reconstructor.get_inputs_and_targets(self.image)[
                1
            ].squeeze()
            probabilities = F.log_softmax(model(self.image, labels).squeeze(), dim=-1)
            unpadded_pbb = probabilities[: self.image_width]

            if pixels_range is not None:
                start, end = pixels_range
                img_pixels_int = img_pixels_int[start:end]
                unpadded_pbb = probabilities[start:end]

            # Reshape to 1D vector
            img_pixels_int_vec = img_pixels_int.reshape(-1)
            unpadded_pbb = unpadded_pbb.reshape(-1, probabilities.size(-1))
            image_pixels_pbb = unpadded_pbb[
                torch.arange(unpadded_pbb.size(0)), img_pixels_int_vec
            ]
            return image_pixels_pbb, probabilities

    def test_model_labeling(
        self,
        model,
        digit_idx,
        num_classes,
        replaced_digit_prob_only=False,
        verbose=True,
    ):
        model.eval()
        testing_info = {}

        digit_idx = min(digit_idx, len(self.digits_positions) - 1)
        assert not self.is_position_ignored(
            digit_idx
        ), f"Symbol at position {digit_idx} is set as ignored."
        digit, start, end = self.digits_positions[digit_idx]
        start, end = start.long(), end.long()
        max_log_prob = -np.inf
        best_digit = None
        labels_log_prob = torch.zeros(num_classes)
        if verbose:
            print(f"Digit with index {digit_idx}: {int(digit)}")
        for i in range(num_classes):
            new_label = i
            labels = self.labels.clone()
            labels[:, start:end] = new_label
            pixels_range = None if not replaced_digit_prob_only else (start, end)
            image_pixels_pbb, probs = self.compute_image_pbb(
                model, labels, pixels_range=pixels_range
            )
            log_prob = torch.sum(image_pixels_pbb)
            labels_log_prob[i] = log_prob
            testing_info[i] = {
                "image_log_pbb": log_prob,
                "image_pixels_pbb": image_pixels_pbb,
                "probabilities": probs,
                "labels": labels,
            }
            if log_prob > max_log_prob:
                max_log_prob, best_digit = log_prob, i
            if verbose:
                print(
                    f"After replecing {int(digit)} with {i} sum of image pixels log probabilities is {log_prob}"
                )
        if verbose:
            print(f"Digit with highest probability ({max_log_prob}): {best_digit}")
        labels_prob = F.softmax(labels_log_prob, dim=-1)
        digit_prob_tuple = (digit.long(), labels_prob[digit.long()], labels_prob)
        return testing_info, digit_prob_tuple

    def plot_image_pixels_pbb(self, model, testing_info):
        img_pixels_int = model.reconstructor.get_inputs_and_targets(self.image)[1]
        for i in testing_info:
            print(
                f"Probabilities of orginal image pixels when labels are set as digit {i}:"
            )
            pixels_score = testing_info[i]["probabilities"][
                : img_pixels_int.size(1), :, :
            ]
            squeezed_pixels_score = pixels_score.view(-1, pixels_score.size(-1))
            image_log_pbb = squeezed_pixels_score[
                torch.arange(squeezed_pixels_score.size(0)), img_pixels_int.view(-1)
            ]
            image_pbb = torch.exp(image_log_pbb.view(self.image_width, -1))

            plt.figure(figsize=(15, 5))
            plt.imshow(1 - image_pbb.data.cpu().transpose(0, 1), cmap="gray")
            plt.show()
