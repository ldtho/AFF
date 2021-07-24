from collections import Counter
import numpy as np
import pickle


def get_num_per_class(labels):
    counts = Counter(labels)
    counts = [(x, counts[x]) for x in counts]
    counts = sorted(counts, key=lambda x: x[0])
    counts = np.array([x[1] for x in counts])
    return counts


def compute_class_balanced_weights(labels, beta: float):
    """
    :param labels: list of labels
    :param beta: make believe hyperparameter
    :return:
    """
    num_per_class = get_num_per_class(labels)
    effective_num = 1.0 - np.power(beta, num_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights)
    return weights


print(compute_class_balanced_weights([1, 0, 0, 0, 0, 0], 0.9))
