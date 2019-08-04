# 自作モデル
from models import load_model
from dataset import load_data

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.preprocessing import image

import os
import json
import numpy as np
from datetime import datetime

np.random.seed(1)
tf.set_random_seed(2)


def main(config):

    # データの読み込み
    x_train, x_test, y_train, y_test = load_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # モデルの読み込み
    model = load_model(config)

    # Tensorboard
    logdir = "log/run-{}/".format(datetime.utcnow().strftime("%Y%m%d%H%M%S"))
    tb_cb = TensorBoard(log_dir=logdir,
                        histogram_freq=1,
                        write_images=1)
    cbks = [tb_cb]

    # augmentation
    datagen = image.ImageDataGenerator(shear_range=5)
    gen = datagen.flow(x_train, y_train, batch_size=config["batch_size"])

    x, y = next(gen)
    print(x.shape, y.shape)
    # 学習の実行
    history = model.fit_generator(gen,
                                  steps_per_epoch=len(x_train)/config["batch_size"],
                                  epochs=config["epochs"],
                                  callbacks=cbks,
                                  validation_data=(x_test, y_test))

    # モデルの構成確認
    model.summary()

    # テスト
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test Accuracy', score[1])

    # モデルの保存
    model.save(os.path.abspath(os.path.dirname(__file__)) + "/../week8/models/katakana_model.h5")


if __name__ == '__main__':
    # parameter
    f = open(os.path.abspath(os.path.dirname(__file__)) + "/config.json")
    config = json.load(f)
    print("Parameters:", config)
    main(config)
