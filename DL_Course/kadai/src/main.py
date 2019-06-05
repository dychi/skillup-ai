import numpy as np


# 自作モデル
from train import train_model
from dataset import preprocess
from model import mymodel







if __name__ == '__main__':
    args = config()
    main(args)