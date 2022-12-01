import matplotlib.pyplot as plt
import scipy.stats
import re
import csv
import numpy as np

data = []

with open("../dataset/VNINDEX_from_01-07-2021_to_31-12-2021.csv") as file:
    reader = csv.reader(file, delimiter=',')
    for row in reader:

        data.append(float(re.findall(r"[-+]?\d*\.\d+|\d+",row[6])[0]))

# plt.hist(data, bins=100, density=True)
# plt.show()

dist = scipy.stats.norm

bounds = [(-100,100), (-100,100)]
res = scipy.stats.fit(dist, data,bounds)
print(res)
res.plot()
plt.show()