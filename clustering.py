from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from matplotlib import pyplot
import numpy
import pickle

dataset = load_digits().data

image_number = 1024

pyplot.gray()
pyplot.matshow(load_digits().images[image_number])

dataset = scale(dataset)

algorithm = pickle.load(open("kmeans.pkl", "rb"))#KMeans(n_clusters=10)
#algorithm.fit(dataset)
#pickle.dump(algorithm, open("kmeans.pkl", "wb"))

mini_batch = numpy.array([dataset[image_number]])
print(algorithm.predict(mini_batch))
