import numpy as np


class BaseEncoder:
    def encode(self, data):
        raise NotImplementedError()

    def decode(self, data):
        raise NotImplementedError()


class OneHot(BaseEncoder):
    """Only for single-label classification"""

    def __init__(self, labels_count):
        self.labels_count = labels_count

    def encode(self, data):
        assert np.max(data) < self.labels_count
        batch_size = data.shape[0]
        result = np.zeros((batch_size, self.labels_count), dtype=int)
        result[range(batch_size), data] = 1
        return result

    def decode(self, data):
        assert data.shape[1] == self.labels_count
        return np.argmax(data, axis=1)
