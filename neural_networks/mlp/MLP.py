import keras
from keras.layers import Input, Dense


class MLP:

    def __init__(self, input_size, neurons_per_layer=[256], activations=['relu']):
        self.input_size = input_size
        self.input_data = Input(shape=(input_size,))
        self.net = self.create_network(neurons_per_layer, activations)
        self.model = keras.Model(self.input_data, self.net)

    def create_network(self, neurons_per_layer, activations):
        net = self.input_data
        for i in range(len(neurons_per_layer)):
            n = neurons_per_layer[i]
            activation = activations[i]
            net = Dense(n, activation)(net)
        return net

    def train(self, X, y, optimizer='adam', loss='binary_crossentropy', batch_size=256, epochs=50):
        self.model.compile(optimizer=optimizer, loss=loss)
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def predict(self, X):
        return self.model.predict(X)
