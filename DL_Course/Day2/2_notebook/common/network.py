import numpy as np
from common.grad import numerical_gradient
from common.activations import softmax, sigmoid
from common.loss import cross_entropy_error

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size):
        
        # 重みの初期化
        self.params = {}
        init_std=0.01
        np.random.seed(1234)
        self.params["W1"] = init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)
                
    def predict(self, x):
        """
        推論関数
        x : 入力データ
        """
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        
        h1 = np.dot(x, W1) + b1
        z1 = sigmoid(h1)
        h2 = np.dot(z1, W2) + b2
        y = softmax(h2)
        return y
    
    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 正解データ
        """
        y = self.predict(x)
        loss = cross_entropy_error(y, t)
        return loss
    
    def gradient(self, x, t):
        """
        勾配計算関数
        """
        def f(W):
            return self.loss(x,t)
        grads={}
        grads["W1"] = numerical_gradient(f, self.params["W1"])
        grads["b1"] = numerical_gradient(f, self.params["b1"])
        grads["W2"] = numerical_gradient(f, self.params["W2"])
        grads["b2"] = numerical_gradient(f, self.params["b2"])
        return grads
