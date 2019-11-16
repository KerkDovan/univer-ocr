from collections import namedtuple


def binary_classification_metrics(prediction, ground_truth, f1beta=1):
    true = (prediction == ground_truth).astype(int)
    false = (prediction != ground_truth).astype(int)
    positives = prediction
    negatives = 1 - prediction
    tp = (true * positives).sum()
    tn = (true * negatives).sum()
    fp = (false * positives).sum()
    fn = (false * negatives).sum()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta2 = f1beta * f1beta
    f1 = (1 + beta2) * precision * recall / (beta2 * precision + recall)
    result = namedtuple(
        'BinaryClassificationMetrics',
        ['accuracy', 'precision', 'recall', 'f1'])
    return result(accuracy, precision, recall, f1)


def multiclass_accuracy(prediction, ground_truth):
    return 0


if __name__ == '__main__':
    import numpy as np
    p = np.array([
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    t = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
    ])
    metrics = binary_classification_metrics(p, t)
    print(metrics)
    print([metrics[i] for i in range(len(metrics))])
    print([x for x in metrics])
    print([getattr(metrics, field) for field in metrics._fields])
    print(metrics._asdict())
