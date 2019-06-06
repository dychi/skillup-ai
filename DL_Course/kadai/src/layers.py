import numpy as np


class ReLU:
    """ReLU関数"""
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy() # 参照渡しではなく、複製
        out[self.mask] = 0
        return out

    def backword(self, dout):
        dout[self.mask] = 0
        dLdx = dout
        return dLdx

