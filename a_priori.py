import itertools

import numpy
import pandas

products = ["eggs", "milk", "bread", "yogurt", "cheese", "pickles", "beer", "water", "diapers", "juice"]

simulated_transactions = 1000000

product_matrix = numpy.zeros((simulated_transactions, len(products)))

for row_number, row in enumerate(product_matrix):
    print(row_number)
    purchases = numpy.random.poisson(4, 7)
    for column in range(0, 10):
        product_matrix[row_number][column] = 1 if column in purchases else 0

numpy.save("sales.npy", product_matrix)

sales = numpy.load("sales.npy")
print(sales[0:10])

sales = pandas.DataFrame(sales)
sales.columns = products

support_n1 = {}
popularity_filter = 0.3

for key in products:
    support_n1[key] = sales[key].sum() / sales[key].shape

filtered_support_n1 = list(filter(lambda tuple: support_n1[tuple] > popularity_filter, support_n1))
print(filtered_support_n1)

support_n2 = {}

tuples = list(itertools.product(filtered_support_n1, filtered_support_n1))
print(len(tuples))
tuples = set(tuple(sorted(l)) for l in tuples)
print(len(tuples))

for pair in tuples:
    if len(set(pair)) > 1:
        support_n2[pair] = sales[(sales[pair[0]] == 1) & (sales[pair[1]] == 1)].shape[0] / sales.shape[0]

print(support_n2)

filtered_support_n2 = list(filter(lambda tuple: support_n2[tuple] > popularity_filter, support_n2))
print(filtered_support_n2)
print(len(filtered_support_n2))
