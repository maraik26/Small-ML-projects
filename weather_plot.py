from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

import csv

spark = SparkSession.builder.master("local").appName("Weather Prediction").config(conf = SparkConf()).getOrCreate()

#Temperature distribution over the entire globe

p = spark.read.csv("Weather_Estimator/tmax-1")

p.show()

l_list = p.select('_c2').collect()
ll_list = p.select('_c3').collect()
t_list = p.select('_c5').collect()

l_array = [float(i._c2) for i in l_list]
ll_array = [float(i._c3) for i in ll_list]
t_array = [float(i._c5) for i in t_list]

map = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90, llcrnrlon=-180,urcrnrlon=180,resolution='c')

fig = plt.figure(figsize=(12, 12))
map.drawmapboundary(fill_color='pink')
map.drawcoastlines()

x, y = map(ll_array, l_array)
plt.title("Temperature over the globe 2000-2016")
map.scatter(x, y, marker='D', s=8, c=t_array)
plt.colorbar(orientation='horizontal')
plt.show()

from pyspark.sql.functions import min, max, abs

p.select(max('_c1')).show()

p.select(min('_c1')).show()

#Temperature distribution over the entire globe

from pyspark.sql.functions import year
tr = p.withColumn("year", year(p._c1))

tr.show()

#a.Temperature distribution over the entire globe for 2012

t12 = tr.filter(tr.year ==2012)

t12.show()

l_list = t12.select('_c2').collect()
ll_list = t12.select('_c3').collect()
t_list = t12.select('_c5').collect() 

l_array = [float(i._c2) for i in l_list]
ll_array = [float(i._c3) for i in ll_list]
t_array = [float(i._c5) for i in t_list]

fig = plt.figure(figsize=(10, 10))
map.drawmapboundary(fill_color='pink')
map.drawcoastlines()

x, y = map(ll_array, l_array)
plt.title("Temperature over the globe 2012")
map.scatter(x, y, marker='D', s=7, c=t_array)
plt.colorbar(orientation='horizontal')
plt.show()

#Temperature distribution over the entire globe for 2009

t16 = tr.filter(tr.year ==2009)

t16.show()

l_list = t16.select('_c2').collect()
ll_list = t16.select('_c3').collect()
t_list = t16.select('_c5').collect() 

l_array = [float(i._c2) for i in l_list]
ll_array = [float(i._c3) for i in ll_list]
t_array = [float(i._c5) for i in t_list]

fig = plt.figure(figsize=(12, 10))
map.drawmapboundary(fill_color='pink')
map.drawcoastlines()


x, y = map(ll_array, l_array)
plt.title("Temperature over the globe 2009")
map.scatter(x, y, marker='D', s=10, c=t_array)
plt.colorbar(orientation='horizontal')
plt.show()

#b1) Evaluate your model using a dense plot

p2 = spark.read.csv("Weather_Estimator/out1")

d_list = diff.select('_c2').collect()
dd_list = diff.select('_c3').collect()
pr_list = diff.select('_c7').collect()

d_array = [float(i._c2) for i in d_list]
dd_array = [float(i._c3) for i in dd_list]
pr_array = [float(i._c7) for i in pr_list]

map = Basemap(projection='cyl', lat_0=0, lon_0=0)

import numpy as np
fig = plt.figure(figsize=(12, 12))
map.drawmapboundary(fill_color='pink')
map.drawcoastlines()

x, y = map(dd_array, d_array)
plt.title("Prediction temperature")
map.scatter(x, y, marker='D', s=10, c=pr_array)
plt.colorbar(orientation='horizontal')
plt.show()

#b2) Regression error of a model prediction vs. test data

p2 = spark.read.csv("Weather_Estimator/out1")
p2.show()
diff = p2.withColumn("Error", abs(p2._c5 - p2._c7))
diff.show()

d_list = diff.select('_c2').collect()
dd_list = diff.select('_c3').collect()
er_list = diff.select('Error').collect()

d_array = [float(i._c2) for i in d_list]
dd_array = [float(i._c3) for i in dd_list]
er_array = [float(i.Error) for i in er_list]

map = Basemap(projection='cyl', lat_0=0, lon_0=0)

fig = plt.figure(figsize=(12, 12))
map.drawmapboundary(fill_color='pink')
map.drawcoastlines()

x, y = map(dd_array, d_array)
plt.title("Regression error of a model prediction vs test data")
map.scatter(x, y, marker='D', s=7, c=er_array)
plt.colorbar(orientation='horizontal')
plt.show()
