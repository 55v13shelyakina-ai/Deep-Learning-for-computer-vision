import torch


def hello():
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)

    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)

    for i in range(num_train):
        for j in range(num_test):
            diff = x_train_flat[i] - x_test_flat[j]
            dists[i, j] = (diff * diff).sum()

    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)

    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)

    for i in range(num_train):
        diff = x_train_flat[i] - x_test_flat
        dists[i] = (diff * diff).sum(dim=1)

    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat  = x_test.reshape(num_test, -1)

    train_sq = (x_train_flat ** 2).sum(dim=1).unsqueeze(1)
    test_sq  = (x_test_flat  ** 2).sum(dim=1).unsqueeze(0)
    cross    = x_train_flat @ x_test_flat.T
    dists = train_sq + test_sq - 2 * cross
    
    return dists


def predict_labels(dists: torch.Tensor, y_train: torch.Tensor, k: int = 1):
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)

    topk_dists, topk_idx = torch.topk(dists, k, dim=0, largest=False)

    for j in range(num_test):
        neighbor_labels = y_train[topk_idx[:, j]]
        counts = torch.bincount(neighbor_labels)
        y_pred[j] = torch.argmax(counts)

    return y_pred


class KnnClassifier:

    def __init__(self, x_train: torch.Tensor, y_train: torch.Tensor):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test: torch.Tensor, k: int = 1):
        y_test_pred = None
        dists = compute_distances_no_loops(self.x_train, x_test)
        y_test_pred = predict_labels(dists, self.y_train, k=k)
        return y_test_pred

    def check_accuracy(
        self,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        k: int = 1,
        quiet: bool = False
    ):
        """
        Utility method for checking the accuracy of this classifier on test
        data. Returns the accuracy of the classifier on the test data, and
        also prints a message giving the accuracy.

        Args:
            x_test: Tensor of shape (num_test, C, H, W) giving test samples.
            y_test: int64 Tensor of shape (num_test,) giving test labels.
            k: The number of neighbors to use for prediction.
            quiet: If True, don't print a message.

        Returns:
            accuracy: Accuracy of this classifier on the test data, as a
                percent. Python float in the range [0, 100]
        """
        y_test_pred = self.predict(x_test, k=k)
        num_samples = x_test.shape[0]
        num_correct = (y_test == y_test_pred).sum().item()
        accuracy = 100.0 * num_correct / num_samples
        msg = (
            f"Got {num_correct} / {num_samples} correct; "
            f"accuracy is {accuracy:.2f}%"
        )
        if not quiet:
            print(msg)
        return accuracy
