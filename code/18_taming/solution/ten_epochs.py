import numpy as np
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.utils import to_categorical
from keras.datasets import mnist

# A utility function that plots the training loss and validation loss from
# a Keras history object.

import matplotlib.pyplot as plt
import seaborn as sns


def plot(history):
    sns.set()  # Switch to the Seaborn look
    plt.plot(history.history['loss'], label='Training set',
             color='blue', linestyle='-')
    plt.plot(history.history['val_loss'], label='Validation set',
             color='green', linestyle='--')
    plt.xlabel("Epochs", fontsize=30)
    plt.ylabel("Loss", fontsize=30)
    plt.xlim(0, len(history.history['loss']))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=30)
    plt.show()


(X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw) = mnist.load_data()
X_train = X_train_raw.reshape(X_train_raw.shape[0], -1) / 255
X_test_all = X_test_raw.reshape(X_test_raw.shape[0], -1) / 255
X_validation, X_test = np.split(X_test_all, 2)
Y_train = to_categorical(Y_train_raw)
Y_validation, Y_test = np.split(to_categorical(Y_test_raw), 2)

model = Sequential()
model.add(Dense(1200, activation='relu'))
model.add(Dropout(0.1))  # dropout is an advanced regularization
model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(200, activation='relu'))
# model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

# model.compile(loss='categorical_crossentropy',  # SGD, learning rate decay,
#               optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9),
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=1e-3),
              metrics=['accuracy'])

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    validation_data=(X_validation, Y_validation),
                    epochs=10, batch_size=32)

plot(history)
