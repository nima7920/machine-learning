import keras
from keras.layers import Dense


class DenseAutoEncoder:

    def __init__(self, input_size, encoding_dims=[2], encoding_activations=['relu'], decoding_dims=[2],
                 decoding_activations=['sigmoid']):
        self.input_size = input_size
        self.input_data = keras.Input(shape=(input_size,))
        self.encoded, self.encoder = self.create_encoder(encoding_dims, encoding_activations)
        self.decoded, self.decoder = self.create_decoder(decoding_dims, decoding_activations)
        self.auto_encoder = keras.Model(self.input_data, self.decoded)

    def create_encoder(self, encoding_dims, encoding_activations):
        encoded = self.input_data
        for i in range(len(encoding_dims)):
            dim = encoding_dims[i]
            activation = encoding_activations[i]
            encoded = Dense(dim, activation)(encoded)
        encoder = keras.Model(self.input_data, encoded)
        return encoded, encoder

    def create_decoder(self, decoding_dims, decoding_activations):
        decoded = self.encoded
        for i in range(1, len(decoding_dims)):
            dim = decoding_dims[i]
            activation = decoding_activations[i]
            decoded = Dense(dim, activation)(decoded)
        decoded = Dense(self.input_size, decoding_activations[-1])(decoded)
        decoder = keras.Model(self.encoded, decoded)
        return decoded, decoder

    def train(self, X,optimizer='adam',loss='binary_crossentropy', batch_size=256, epochs=50):
        self.auto_encoder.compile(optimizer=optimizer,loss=loss)
        self.auto_encoder.fit(X, X, batch_size, epochs,verbose=False)

    def predict(self, X):
        return self.auto_encoder.predict(X)

    def encode(self,X):
        return self.encoder.predict(X)

