import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x,w) + b
        y_hat = 1 / (1 + np.exp(-z))
        err = (y_hat - y_true) * y_hat * (1- y_hat)
        m = w.shape[0]
        dl_dw = np.zeros(m)
        for j in range(m):
            dl_dw[j] = err * x[j]
        dl_db = err
        return (np.round(dl_dw,5),round(dl_db,5))

