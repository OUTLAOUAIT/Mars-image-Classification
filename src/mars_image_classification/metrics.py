"""Small dependency-light classification metrics."""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    if np.any((true < 0) | (true >= num_classes) | (pred < 0) | (pred >= num_classes)):
        raise ValueError("Class index outside configured range")
    np.add.at(matrix, (true, pred), 1)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    denom = support + predicted
    f1 = np.divide(2 * np.diag(matrix), denom, out=np.zeros(num_classes, dtype=float), where=denom != 0)
    return {
        "accuracy": float(np.trace(matrix) / true.size),
        "macro_f1": float(f1.mean()),
        "confusion_matrix": matrix.tolist(),
        "support": support.tolist(),
    }
