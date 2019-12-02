from .gpu import CP


class BaseLoss:
    def __call__(self, prediction, ground_truth):
        raise NotImplementedError()


class SegmentationDice2D(BaseLoss):
    """https://www.jeremyjordan.me/semantic-segmentation/#loss"""

    def __call__(self, prediction, ground_truth):
        batch_size, _, _, channels = prediction.shape
        new_shape = (batch_size, 1, 1, channels)

        def sum_reshape(array):
            return CP.cp.sum(array, axis=(1, 2)).reshape(new_shape)

        numerator = sum_reshape(prediction * ground_truth)
        denominator = sum_reshape(prediction) + sum_reshape(ground_truth)

        loss = CP.cp.sum(1 - 2 * numerator / denominator)
        grad = -2 * (ground_truth * denominator - numerator) / denominator ** 2
        return float(loss), grad


class SegmentationJaccard2D(BaseLoss):
    def __call__(self, prediction, ground_truth):
        batch_size, _, _, channels = prediction.shape
        new_shape = (batch_size, 1, 1, channels)

        def sum_reshape(array):
            return CP.cp.sum(array, axis=(1, 2)).reshape(new_shape)

        numerator = sum_reshape(prediction * ground_truth)
        denominator = sum_reshape(prediction) + sum_reshape(ground_truth) - numerator

        loss = CP.cp.sum(1 - numerator / denominator)
        grad = -(ground_truth * denominator - numerator * (1 - ground_truth)) / denominator ** 2
        return float(loss), grad


class SigmoidCrossEntropy(BaseLoss):
    """https://gombru.github.io/2018/05/23/cross_entropy_loss/"""

    def __call__(self, prediction, ground_truth):
        def sigmoid(X):
            return 1 / (1 + CP.cp.exp(-X))
        batch_size = ground_truth.shape[0]
        pred = sigmoid(prediction)
        gt, rev_gt = ground_truth, 1 - ground_truth
        loss = -(CP.cp.sum((gt * CP.cp.log(pred)) + rev_gt * CP.cp.log(1 - pred))
                 ) / batch_size
        grad = ((gt * (pred - 1)) + rev_gt * pred) / batch_size
        return float(loss), grad


class SoftmaxCrossEntropy(BaseLoss):
    """https://deepnotes.io/softmax-crossentropy"""

    def __call__(self, prediction, ground_truth):
        def softmax(X):
            eX = CP.cp.exp(X - CP.cp.max(X))
            seX = CP.cp.sum(eX, axis=1)
            seX = CP.cp.reshape(seX, (*seX.shape, 1))
            return eX / seX
        batch_size = ground_truth.shape[0]
        predicted = softmax(prediction)
        loss = -(CP.cp.sum(ground_truth * CP.cp.log(predicted))) / batch_size
        grad = (predicted - ground_truth) / batch_size
        return float(loss), grad
