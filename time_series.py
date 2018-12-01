from datetime import datetime

import numpy
import pandas
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def date_formatter(row):
    return datetime.strptime(row, "%Y-%m-%d")

dataframe = pandas.read_csv("walmart.csv",
                            parse_dates=["fecha2"],
                            date_parser=date_formatter,
                            index_col=2)

dataframe.columns = ["store", "t"]
dataframe.index.name = "date"
dataframe = dataframe[dataframe["store"] == 58].drop(labels=["store"], axis=1)
dataframe["t+3"] = dataframe["t"].shift(-3)
dataframe["t+2"] = dataframe["t"].shift(-2)
dataframe["t+1"] = dataframe["t"].shift(-1)
dataframe["t-3"] = dataframe["t"].shift(3)
dataframe["t-2"] = dataframe["t"].shift(2)
dataframe["t-1"] = dataframe["t"].shift(1)
dataframe = dataframe.dropna().values.astype("float32")

#scaler = MinMaxScaler(feature_range=(0, 1))
#dataframe = scaler.fit_transform(dataframe)
universal_max = numpy.max(dataframe[:,0])
dataframe /= universal_max

train_dataset = dataframe[:int(dataframe.shape[0] * 0.8)]
test_dataset = dataframe[int(dataframe.shape[0] * 0.8):]

train_X, train_y = train_dataset[:, 4:], train_dataset[:, :4]
test_X, test_y = test_dataset[:, 4:], test_dataset[:, :4]

train_X = train_X.reshape((train_X.shape[0], train_X.shape[1], 1))
test_X = test_X.reshape((test_X.shape[0], test_X.shape[1], 1))

model = Sequential()
lstm = LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]))
model.add(lstm)
dense = Dense(4)
model.add(dense)

model.compile(loss="mae", optimizer="adam", metrics=["mse"])

model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y))
model.save("timeSeries.h5")

model = load_model("timeSeries.h5")
model.compile(loss="mae", optimizer="adam", metrics=["mse"])

'''
model.fit(train_X, train_y, epochs=100, batch_size=32, validation_data=(test_X, test_y))
model.save("timeSeries.h5")
'''

X = test_X[21]
print(X)

accuracy = model.evaluate(test_X, test_y)
print(accuracy)

prediction = model.predict(numpy.array([X]))
print(prediction * universal_max, test_y[21] * universal_max)
