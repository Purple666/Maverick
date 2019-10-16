from itertools import product
import random

from keras.models import Sequential
from keras.layers import LSTM, SpatialDropout1D, Bidirectional
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from bayes_opt import BayesianOptimization

class Maverick:
    def __init__(self, x_train, y_train, validation_data=None, callbacks=True):
        self.x_train = x_train
        self.y_train = y_train
        self.validation_data = validation_data
        self.best_final= {'val_loss': None, 'val_acc': None, 'params': None}
        self.best= {'val_loss': None, 'val_acc': None, 'params': None}
        self.worst= {'val_loss': None, 'val_acc': None, 'params': None}
        self._callbacks = [EarlyStopping(monitor='val_acc', patience=20, restore_best_weights=True)] if callbacks else None
        self._model = None

    def run_model(self, batch_size, drop_out, lr, n1, n2, n3):
        self._model = Sequential()
        self._model.add(Bidirectional(LSTM(n1, return_sequences=True, input_shape=(self.x_train.shape[1], self.x_train.shape[2]))))
        self._model.add(SpatialDropout1D(drop_out))
        self._model.add(Bidirectional(LSTM(n2, return_sequences=True)))
        self._model.add(SpatialDropout1D(drop_out))
        self._model.add(Bidirectional(LSTM(n3, return_sequences=False)))
        self._model.add(Dense(self.y_train.shape[1], activation="softmax"))
        self._model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr), metrics=['accuracy'])
        return self._model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=1000, validation_split=0.2, validation_data=self.validation_data, verbose=1, callbacks=self._callbacks)
    
    def predict(self, x):
        return self._model.predict(x)

    def grid_search(self, random_shuffle=False, batch_size=None, drop_out=None, learning_rate=None, neurons=None):
        params = list(product(batch_size, drop_out, learning_rate, *neurons))
        if random_shuffle:
            random.shuffle(params)
        for batch_size, drop_out, lr, n1, n2, n3 in params:
            print("next:", batch_size, drop_out, lr, [n1, n2, n3])
            history = self.run_model(batch_size, drop_out, lr, n1, n2, n3)

            result = {
                'val_loss': history.history['val_loss'][-1],
                'val_acc': history.history['val_acc'][-1],
                'params': {'batch_size':batch_size, 'drop_out': drop_out, 'learning_rate': lr, 'neurons': [n1, n2, n3]}
            }
            if not self.best_final['val_loss'] or self.best_final['val_loss'] > history.history['val_loss'][-1]:
                self.best_final.update(result)
            if not self.worst['val_loss'] or self.worst['val_loss'] < history.history['val_loss'][-1]:
                self.worst.update(result)
            if not self.best['val_loss'] or self.best['val_loss'] > min(history.history['val_loss']):
                result.update({
                    'val_loss': min(history.history['val_loss']),
                    'val_acc': max(history.history['val_acc'])
                })
                self.best.update(result)
            print("current:", batch_size, drop_out, lr, [n1, n2, n3])
            print("worst:", self.worst)
            print("best final:", self.best_final)
            print("best:", self.best)

    def bayesian_optimization(self, batch_size=None, drop_out=None, learning_rate=None, neurons=None):
        def black_box(batch_size, drop_out, lr, n1, n2, n3):
            batch_size = int(batch_size)
            n1 = int(n1)
            n2 = int(n2)
            n3 = int(n3)
            history = self.run_model(batch_size, drop_out, lr, n1, n2, n3)
            return max(history.history['val_acc'])
        pbounds = {
            'batch_size': batch_size,
            'drop_out': drop_out,
            'lr': learning_rate,
            'n1': neurons[0],
            'n2': neurons[1],
            'n3': neurons[2]
        }
        optimizer = BayesianOptimization(
            f=black_box,
            pbounds=pbounds,
            random_state=1,
        )
        optimizer.maximize(init_points=10, n_iter=50)
        print(optimizer.max)
        return optimizer.max