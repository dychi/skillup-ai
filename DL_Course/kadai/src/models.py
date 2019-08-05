from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation, Dropout
from keras.models import Model
from keras.optimizers import Adam


def load_model(config):
    input = Input(shape=(28,28,1))
    # layer1
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(input)
    x = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    # layer2
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(x)
    x = Dropout(0.25)(x)
    # layer3
    x = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    out = Dense(15, activation='softmax')(x)

    model = Model(inputs=input, outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=config["lr"], epsilon=1e-08),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    from keras.utils import plot_model
    model = load_model()
    # plot_model(model, to_file='model.png')
    model.summary()
