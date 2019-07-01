from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam

import config


def load_model():
    input = Input(shape=(28,28,1))
    # layer1
    x = Conv2D(64, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='valid')(x)
    # layer2
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='valid')(x)
    # layer3
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='valid')(x)

    x = Flatten()(x)

    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(64, activation='relu')(x)

    out = Dense(15, activation='softmax')(x)

    model = Model(inputs=input, outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=config.lr),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    from keras.utils import plot_model
    model = load_model()
    # plot_model(model, to_file='model.png')
    model.summary()
