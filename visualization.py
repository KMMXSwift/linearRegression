import pandas
import matplotlib.pyplot
import seaborn

iris = seaborn.load_dataset("tips")

seaborn.set()

'''
plot = seaborn.swarmplot(x="species", y="sepal_length", data=iris).get_figure()

plot.savefig("swarmplot.png")
'''

'''
species = iris.pop("species")

clusters = seaborn.clustermap(iris, metric="jaccard")

clusters.savefig("clustermap.png")
'''

'''
graph = seaborn.catplot(x="day", y="total_bill", hue="sex", data=iris, kind="swarm")
graph.savefig("catplot.png")
'''

'''
graph = seaborn.catplot(x="smoker", y="tip", order=["No", "Yes"], hue="sex", data=iris)
graph.savefig("catplot.png")
'''

'''
graph = seaborn.catplot(x="total_bill", y="day", hue="time", kind="swarm", data=iris)
graph.savefig("catplot.png")
'''

graph = seaborn.catplot(x="day", y="total_bill", hue="smoker", kind="box", data=iris)
graph.savefig("catplot.png")