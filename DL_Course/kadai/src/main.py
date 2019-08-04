# 自作モデル
from models import load_model
from dataset import load_data
from keras.callbacks import TensorBoard
from keras.preprocessing import image
import tensorflow as tf


import os
import json
import numpy as np

np.random.seed(1)
tf.set_random_seed(2)


def main(config):

    # モデルの読み込み
    model = load_model(config)

    # データの読み込み
    x_train, x_test, y_train, y_test = load_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # 前処理
    datagen = image.ImageDataGenerator(
        shear_range=5
    )
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train, config['batch_size']),
                        steps_per_epoch=len(x_train)/config['batch_size'],
                        epochs=config['epochs'])

    for e in range(config['epochs']):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=config['batch_size']):
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train)/config['batch_size']:
                break
    # Tensorboard
    tb_cb = TensorBoard(log_dir='./tflog/')
    cbks = [tb_cb]

    # 学習の実行
    history = model.fit(x_train, y_train,
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        callbacks=cbks,
                        validation_data=(x_test, y_test))

    # モデルの構成確認
    model.summary()

    # テスト
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test Loss:', score[0])
    print('Test Accuracy', score[1])

    # モデルの保存
    model.save('/Users/daichi/Ledge/skillup-ai/DL_Course/kadai/week8/models/katakana_model.h5')


if __name__ == '__main__':
    # パラメータの読み込み
    path = os.path.abspath(os.path.dirname(__file__))
    print(path)
    f = open(path + "/config.json", "r")
    config = json.load(f)
    print("Parameters:", config)
    main(config)
