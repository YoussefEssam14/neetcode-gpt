import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        h = x
        n = len(weights)
        for i in range(n):
            h = h @ weights[i] + biases[i]
            if i < n - 1:
                h = np.maximum(0,h)
        return np.round(h,5)
