import numpy as np
import torch.nn.functional as F

from distsup.reconstructors_analysis.style_modeling import single_author, qmnist


class ModelGeneratingTester:
    def __init__(
        self,
        trainer: single_author.trainer.SingleAuthorDigitsTrainer,
        dataset: qmnist.dataset.QMNISTSingleAuthorSet,
    ):
        self.trainer = trainer
        self.dataset = dataset

    def test_model(self, num_authors=30, num_digit_samples=1):
        authors = np.random.choice(len(self.dataset), num_authors, replace=False)
        test_results = []
        for author in authors:
            for i in range(10):
                for j in range(num_digit_samples):
                    sample, log_pbbs = self.trainer.test_generated_sample_class_pbb(
                        self.dataset[author], class_label=i
                    )
                    highest_prob = max(log_pbbs.items(), key=lambda x: x[1])[0]
                    digits_pbbs = F.softmax(torch.tensor(list(log_pbbs.values())))
                    test_entry = {
                        "author_id": author,
                        "digit": i,
                        "sample": sample,
                        "probabilities": digits_pbbs,
                        "true_digit_pbb": digits_pbbs[i],
                        "digit_log_pbbs": log_pbbs,
                        "highest_prob": highest_prob,
                        "correctly_classified": highest_prob == i,
                    }
                    test_results.append(test_entry)
        return test_results
