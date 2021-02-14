# A four-layered neural network.

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.utils import to_categorical
import echidna as data
import boundary

X_train = data.X_train
X_validation = data.X_validation
Y_train = to_categorical(data.Y_train)  # one-hot encode the labels with keras's to_categorical()
Y_validation = to_categorical(data.Y_validation)  # behave the same as the one_hot_encode() in Part I

# define the shape of the NN
model = Sequential()  # no need input layer, so have 5 layers of the NN
# model.add(Dense(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid')) # first hidden layer
model.add(Dense(30, activation='sigmoid'))
model.add(Dense(2, activation='softmax')) # output layer

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

model.fit(X_train, Y_train,  # train the NN
          validation_data=(X_validation, Y_validation),
          epochs=30000, batch_size=25)

boundary.show(model, data.X_train, data.Y_train)
