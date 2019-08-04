from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Activation
from keras.models import Model
from keras.optimizers import Adam


def load_model(args):
    input = Input(shape=(28,28,1))
    x = Conv2D(64, (3, 3), padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='valid')(x)
    x = Conv2D(64, (3, 3))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(1, 1), padding='valid')(x)

    x = Flatten()(x)

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    out = Dense(15, activation='softmax')(x)

    model = Model(inputs=input, outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=args['lr']),
                  metrics=['accuracy'])
    return model


if __name__ == '__main__':
    from keras.utils import plot_model
    model = load_model()
    # plot_model(model, to_file='model.png')
    model.summary()
