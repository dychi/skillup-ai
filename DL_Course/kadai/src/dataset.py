import numpy as np



# データの読み込み
train_data = np.load('../Day1/1_data/train_data.npy')
train_label = np.load('../Day1/1_data/train_label.npy')
print("train data shape:", train_data.shape)
print("train label shape:", train_label.shape)

# 正規化

