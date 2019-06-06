# 自作モデル
from models import load_model
from dataset import load_data
import config


def main():

    # データの読み込み
    X_train, X_test, y_train, y_test = load_data()
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # モデルの読み込み
    model = load_model()

    # 学習の実行
    history = model.fit(X_train, y_train,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        validation_data=(X_test, y_test))

    # モデルの構成確認
    model.summary()

    # テスト
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test Loss:', score[0])
    print('Test Accuracy', score[1])

if __name__ == '__main__':
    # args = config()
    main()
