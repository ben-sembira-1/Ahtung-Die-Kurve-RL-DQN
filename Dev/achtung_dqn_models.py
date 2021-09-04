from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Flatten, Convolution2D, Conv2D, MaxPool2D, concatenate
from keras.engine.functional import Functional


class AchtungDQNModel:
    def __init__(self, input_shape, output_n, verbose=True):
        self.model = self._generate_model(input_shape, output_n)
        if verbose:
            print(self.model.summary())

    def _generate_model(self, input_shape, output_n) -> Functional:
        # A user should override.
        pass


# Working model!
class ConvDense512(AchtungDQNModel):
    def __init__(self, input_shape, output_n, verbose=True):
        super(ConvDense512, self).__init__(input_shape, output_n, verbose)

    def _generate_model(self, input_shape, output_n) -> Functional:
        main_input = Input(input_shape)
        hidden = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(main_input)
        hidden = MaxPool2D((2, 2), padding='same')(hidden)
        hidden = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
        hidden = MaxPool2D((2, 2), padding='same')(hidden)
        hidden = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
        hidden = MaxPool2D((2, 2), padding='same')(hidden)
        hidden = Flatten()(hidden)
        hidden = Dense(512, activation='relu')(hidden)
        main_output = Dense(output_n, activation='relu')(hidden)
        model = Model(inputs=[main_input], outputs=[main_output])
        return model


class DenseDecreasingPowersOf2(AchtungDQNModel):
    def __init__(self, input_shape, output_n, verbose=True):
        super(DenseDecreasingPowersOf2, self).__init__(input_shape, output_n, verbose)

    def _generate_model(self, input_shape, output_n) -> Functional:
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(Activation('relu'))
        model.add(Dense(output_n))
        model.add(Activation('relu'))
        return model


# Todo:
# def make_conv_layer(input):
#     hidden = Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same')(input)
#     hidden = MaxPool2D((2, 2), padding='same')(hidden)
#     hidden = Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
#     hidden = MaxPool2D((2, 2), padding='same')(hidden)
#     hidden = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(hidden)
#     hidden = MaxPool2D((2, 2), padding='same')(hidden)
#     return hidden
#
# input1 = Input((1,) + env.observation_space[0].shape)
# conv1 = make_conv_layer(input1)
# input2 = Input((1,) + env.observation_space[1].shape)
# conv2 = make_conv_layer(input2)
#
# conv = concatenate([conv1, conv2])
#
# hidden = Dense(512, activation='relu')(conv)
# hidden = Dense(nb_actions, activation='relu')(hidden)
# hidden = Dense(256, activation='relu')(hidden)
# hidden = Dense(nb_actions, activation='relu')(hidden)
# hidden = Dense(64, activation='relu')(hidden)
# hidden = Dense(nb_actions, activation='relu')(hidden)
# hidden = Dense(16, activation='relu')(hidden)
# main_output = Dense(nb_actions, activation='relu')(hidden)
# model = Model(inputs=[input1, input2], outputs=[main_output])
#
# model.summary()


# model = Sequential()
# model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(3, 3), input_shape=env.observation_space.shape)) # if soesn't work, consider adding (1,) to the input shape
# model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))
# model.add(Convolution2D(32, 3, 3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dense(nb_actions, activation = 'relu'))
# print(model.summary())


'''
another possible model, if the above to slow or that the subsampling makes problems:

model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=env.observation_space.shape)) # if doesn't work, consider adding (1,) to the input shape
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_actions, activation = 'relu'))
'''
# --------------------------------------------------------------------
