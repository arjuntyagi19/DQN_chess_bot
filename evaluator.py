import numpy as np
import random
import chess
import board

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import layers

tf.compat.v1.disable_eager_execution() # otherwise it is slow !!!


class evaluator_class():
    def __init__(self):

        self.value = Sequential()
        self.value.add(layers.Conv2D(128, (1, 1), activation='relu', input_shape=(8, 8, 26)))
        self.value.add(layers.Conv2D(32, (1, 1), activation='relu'))
        self.value.add(layers.Conv2D(32, (8, 8), groups=32, activation='relu'))
        self.value.add(layers.Flatten())
        self.value.add(layers.Dense(128, activation = "relu"))
        self.value.add(layers.Dense(32, activation = "relu"))
        self.value.add(layers.Dense(1,   activation = "tanh"))
        
        self.value.compile(optimizer= tf.keras.optimizers.legacy.Adam(
            learning_rate = 0.0001), loss='mse')
                              

    def eval(self, b):

        if b.turn == chess.BLACK:
            c = b.mirror()
            return -self.eval(c)

        terminal, score = board.is_terminal(b)
        if terminal:
            return score

        features = board.board_to_features(b)
        
        # print(features.shape)

        f2 = np.zeros((1,8,8,26))
        f2[0] = features
        
        score = self.value.predict(f2)[0][0]

        return np.clip(score, -1, 1)

            
            



        

        
        
