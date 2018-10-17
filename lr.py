import pandas
import seaborn
import numpy
import scipy

dataframe = pandas.read_csv("day.csv")
print(dataframe.info())
print(dataframe.describe())
print(dataframe.head(3))
print(dataframe.tail(3))

lr_data = dataframe[["temp", "cnt"]]
print(lr_data.head())

'''
def power_2(x):
    return x ** 2

lr_data["cnt"] = lr_data["cnt"].apply(power_2)
print(lr_data.head())
'''

plot = seaborn.lmplot("temp", "cnt", data=lr_data)
plot.savefig("figure.png", dpi=300)

is_correlated = numpy.corrcoef(lr_data["temp"], lr_data["cnt"])
print(is_correlated)

def gradient_descent(s_slope, s_intercept, l_rate, iter_val, x_train, y_train):
    for i in range(iter_val):
        int_slope = 0
        int_intercept = 0

        n_pt = float(len(x_train))

        y_hat = (s_slope * x_train[i]) + s_intercept

        for i in range(len(x_train)):
            int_intercept = - (2/n_pt) * (y_train[i] - y_hat)
            int_slope = - (2/n_pt) * x_train[i] * (y_train[i] - y_hat)

        s_slope = s_slope - (l_rate * int_slope)
        s_intercept = s_intercept - (l_rate * int_intercept)

        y_hat = (s_slope * x_train[0]) + s_intercept
        print("Y Train: {}".format(y_train[0]))
        print("Y Hat: {}".format(y_hat))

    return s_slope, s_intercept

s_slope = 1
s_intercept = 1

x_train = lr_data["temp"].as_matrix()
y_train = lr_data["cnt"].as_matrix()

y_hat = (s_slope * x_train[0]) + s_intercept
print("Y Train: {}".format(y_train[0]))
print("Y Hat: {}".format(y_hat))

s_slope, s_intercept = gradient_descent(s_slope, s_intercept, 0.7, 800, x_train, y_train)
print(s_slope, s_intercept)

slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_train, y_train)
print(slope, intercept)
