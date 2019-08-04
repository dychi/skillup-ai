# coding: utf-8
import numpy as np
from common.layers import * 
from common.functions import sigmoid


class RNN:
    def __init__(self, Wx, Wh, b):
        
        # パラメータのリスト
        self.params = [Wx, Wh, b]
        
        # 勾配のリスト
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """
        順伝播計算
        """
        Wx, Wh, b = self.params
        
        # 行列の積　+　行列の積 + バイアス
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        
        # 活性化関数に入れる
        h_next = np.tanh(t)

        # 値の一時保存
        self.cache = (x, h_prev, h_next)
        
        return h_next

    def backward(self, dh_next):
        """
        逆伝播計算
        """
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        # tanhでの逆伝播
        # dh_next * (1 - y^2)
        A3 = dh_next * (1 - h_next ** 2)
        
        # バイアスbの勾配
        # Nの方向に合計する
        db = np.sum(A3, axis=0)
        
        # 重みWhの勾配
        dWh = np.dot(h_prev.T, A3)
        
        # 1時刻前に渡す勾配
        dh_prev = np.dot(A3, Wh.T)
        
        # 重みWxの勾配
        dWx = np.dot(x.T, A3)
        
        # 入力xに渡す勾配
        dx = np.dot(A3, Wx.T)

        # 勾配をまとめる
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db        

        return dx, dh_prev


class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

        
class Embedding:
    def __init__(self, W):
        """
        W : 重み行列, word2vecの埋め込み行列に相当する
        """
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        """
        順伝播計算
        """
        W, = self.params
        self.idx = idx
        
        # 埋め込み行列から埋め込みベクトルを取り出す
        out = W[idx]
        
        return out

    def backward(self, dout):
        """
        逆伝播計算
        """
        dW, = self.grads
        
        # 配列の全ての要素に0を代入する
        dW[...] = 0
        
        # dWのidxの場所にdoutを代入する
        np.add.at(dW, self.idx, dout)
        return None

    
class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        
        # パラメータのリスト
        self.params = [W, b]
        
        # 勾配のリスト
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        
        self.x = None

    def forward(self, x):
        """
        順伝播計算
        x : 入力データ
        """
        N, T, D = x.shape # ミニバッチ数、時間数、前層のノード数
        W, b = self.params

        # 全ての時刻について、一度でAffineの順伝播計算を行う
        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b # 行列の積 + バイアス
        
        # xを保持
        self.x = x
        
        return out.reshape(N, T, -1)

    def backward(self, dout):
        """
        逆伝播計算
        """
        x = self.x
        N, T, D = x.shape # ミニバッチ数、時間数、前層のノード数
        W, b = self.params

        # 全ての時刻について、一度でAffineの逆伝播計算を行う
        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)
        db = np.sum(dout, axis=0) # バイアスの勾配
        dW = np.dot(rx.T, dout) # 重みWの勾配
        dx = np.dot(dout, W.T) # 前層へ伝える勾配
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 教師ラベルがone-hotベクトルの場合
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # バッチ分と時系列分をまとめる（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # ignore_labelに該当するデータは損失を0にする
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # ignore_labelに該当するデータは勾配を0にする

        dx = dx.reshape((N, T, V))

        return dx

class LSTM:
    def __init__(self, Wx, Wh, b):
        '''
        Parameters
        ----------
        Wx: 入力x用の重みパラーメタ（4つ分の重みをまとめたもの)
        Wh: 隠れ状態h用の重みパラメータ（4つ分の重みをまとめたもの）
        b: バイアス（4つ分のバイアスをまとめたもの）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        """
        順伝播計算
        """        
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]

        f = sigmoid(f)
        g = np.tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
    
        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        """
        順伝播計算
        """        
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev




class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

        
        
class GRU:
    def __init__(self, Wx, Wh, b):
        '''
        Wx: 入力x用の重みパラーメタ（3つ分の重みをまとめたもの）
        Wh: 隠れ状態h用の重みパラメータ（3つ分の重みをまとめたもの）
        b: バイアス（3つ分のバイアスをまとめたもの）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """
        順伝播計算
        """
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bhz,   bhr,  bhh =  b[:H], b[H:2 * H], b[2 * H:]
        
        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bhz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + bhr)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r*h_prev, Whh) + bhh)
        h_next = z * h_prev + (1-z) * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        """
        逆伝播計算
        """        
        Wx, Wh, b = self.params
    
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat = dh_next * (1 - z)
        dh_prev = dh_next * z

        # tanh
        dt = dh_hat * (1 - h_hat ** 2)
        dbt = dt
        dWhh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Whh.T)
        dWxh = np.dot(x.T, dt)
        dx = np.dot(dt, Wxh.T)
        dh_prev += r * dhr

        # update gate(z)
        dz =  dh_next * h_prev - dh_next * h_hat
        dt = dz * z * (1-z)
        dbz = dt
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # reset gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1-r)
        dbr = dt
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        dA = np.hstack((dbz, dbr, dbt ))
        
        dWx = np.hstack((dWxz, dWxr, dWxh))
        dWh = np.hstack((dWhz, dWhr, dWhh))
        db = dA.sum(axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        return dx, dh_prev
    

class TimeGRU:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]        
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H, H3 = Wh.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = GRU(Wx, Wh, b)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh= 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None    