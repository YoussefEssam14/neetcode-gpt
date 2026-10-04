import numpy as np
from numpy.typing import NDArray

def sigmoid(z : float):
    return 1 / (1 + np.exp(-z))
def Relu(z : float):
    return max(0.0,z)
class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.dot(x,w) + b
        ans = sigmoid(z) if activation == "sigmoid" else Relu(z)
        return round(ans,5) 