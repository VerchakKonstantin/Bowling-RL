from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras import backend as K


def get_model(input_shape, actions_available):
    # Input layer
    inputs = Input(shape=input_shape)

    # Convolutions on the frames on the screen
    layer1 = Conv2D(16, 8, strides=4, activation="relu")(inputs)
    layer2 = Conv2D(32, 4, strides=2, activation="relu")(layer1)

    # Flatten and add densely connected layer
    layer3 = Flatten()(layer2)
    layer4 = Dense(256, activation="relu")(layer3)

    # Actor and critic layers
    actor = Dense(actions_available, activation="softmax")(layer4)
    critic = Dense(1, activation="linear")(layer4)

    return Model(inputs=inputs, outputs=[actor, critic])


def compute_total_loss(actor_loss, critic_loss, entropy_bonus, c1_coeff, c2_coeff):
    total_loss = actor_loss - (c1_coeff * critic_loss) + (c2_coeff * entropy_bonus)
    return K.mean(total_loss)
