import keras
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
conv = Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1))
model.add(conv)

conv_2 = Conv2D(64, (3, 3), activation="relu")
model.add(conv_2)

pool = MaxPooling2D((2, 2))
model.add(pool)

dropout = Dropout(0.25)
model.add(dropout)

flatten = Flatten()
model.add(flatten)

dense = Dense(128, activation="relu")
model.add(dense)

dropout_2 = Dropout(0.5)
model.add(dropout_2)

dense_2 = Dense(10, activation="softmax")
model.add(dense_2)

model.compile(loss="categorical_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])

model.fit(x_train,
          y_train,
          batch_size=32,
          epochs=3,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test)
print(score)
