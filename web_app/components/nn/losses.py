import numpy as np


class BaseLoss:
    def __call__(self, prediction, ground_truth):
        raise NotImplementedError()


class SigmoidCrossEntropy(BaseLoss):
    """https://gombru.github.io/2018/05/23/cross_entropy_loss/"""

    def __call__(self, prediction, ground_truth):
        def sigmoid(X):
            return 1 / (1 + np.exp(-X))
        batch_size = ground_truth.shape[0]
        pred = sigmoid(prediction)
        gt, rev_gt = ground_truth, 1 - ground_truth
        loss = -(np.sum((gt * np.log(pred)) + rev_gt * np.log(1 - pred))
                 ) / batch_size
        grad = ((gt * (pred - 1)) + rev_gt * pred) / batch_size
        return loss, grad


class SoftmaxCrossEntropy(BaseLoss):
    """https://deepnotes.io/softmax-crossentropy"""

    def __call__(self, prediction, ground_truth):
        def softmax(X):
            eX = np.exp(X - np.max(X))
            seX = np.transpose([np.sum(eX, axis=1)])
            return eX / seX
        batch_size = ground_truth.shape[0]
        predicted = softmax(prediction)
        loss = -(np.sum(ground_truth * np.log(predicted))) / batch_size
        grad = (predicted - ground_truth) / batch_size
        return loss, grad
