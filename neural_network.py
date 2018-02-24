from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from neural_network_utils import auc


def create_nn():
    """Create a simple neural network"""
    
    model = Sequential()
    model.add(Dense(100, input_shape=(100,), activation='relu'))
    model.add(Dense(100, input_shape=(100,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model