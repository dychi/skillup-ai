import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


# データの読み込み
train_data = np.load('./data/train_data.npy')
train_label = np.load('./data/train_label.npy')
print("train data shape:", train_data.shape)
print("train label shape:", train_label.shape)

# 正規化
