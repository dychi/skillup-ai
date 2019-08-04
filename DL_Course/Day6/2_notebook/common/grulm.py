# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
#from common.np import *  # 
import numpy as np
from common.base_model import BaseModel


class GRUlm(BaseModel):
    '''
     GRUレイヤを利用した言語モデル
    '''    
    def __init__(self, vocab_size=10000, wordvec_size=650,
                 hidden_size=650):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 3 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 3 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(3 * H).astype('f')      
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeGRU(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeAffine(affine_W, affine_b) 
        ]
        self.loss_layer = TimeSoftmaxWithLoss()

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
