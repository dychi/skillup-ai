import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data():
    # パスを指定
    DATA_ROOT = Path(os.path.dirname(__file__), '../template/1_data')

    # データの読み込み
    train_data = np.load(DATA_ROOT.joinpath('train_data.npy'))
    train_label = np.load(DATA_ROOT.joinpath('train_label.npy'))
    train_data = train_data.transpose(0, 2, 3, 1)
    print("train data shape:", train_data.shape)
    print("train label shape:", train_label.shape)

    # 前処理
    X_train, X_test, y_train, y_test = preprocess(train_data, train_label)

    return X_train, X_test, y_train, y_test


def preprocess(train_data, train_label):
    # 正規化
    train_data = (train_data - train_data.min()) / train_data.max()
    train_data = train_data.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.2,
                                                        random_state=1234, shuffle=True)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    load_data()
