#%%
import pandas as pd
import plotly.express as px
import hvplot.pandas
from sklearn.cluster import KMeans

#%%
# Loading file
filePath = "data/iris_clean.csv"
irisDF = pd.read_csv(filePath)
irisDF.head()

# %%
# Initially we do not know how many clusters we will need 
# so we start with trial and error

# Create model instance:
# Initializing model with K=3 (since we already know that there are three clases of iris plants)
model = KMeans(n_clusters=3, random_state=5)
model

# %%
# Fitting the model:
# (the K-means algorithm will iteratively look for the best centroid for each of the K clusters)
model.fit(irisDF)

#%%
# Get Predictions:
predictions = model.predict(irisDF)
print(predictions)
# => There were three subclasses that were labeled 0, 1, and 2. These are not the means for the 
#   centroids, but rather just the label names. The actual naming of the classes is part of the 
#   job by a subject matter expert, or whoever performs the analysis. 
#   The K-means algorithm is able to identify how many clusters are in the data and label them with 
#   numbers.

# %%
# Add a new column to the iris DataFrame with the predicted classes
irisDF['class'] = model.labels_
irisDF.head()

# %%
# Plotting the clusters with two features
irisDF.hvplot.scatter(x="sepal_length", y="sepal_width", by="class")
# This plot does not clearly separates each class as we expected

# %%
# Plotting the clusters with three features
fig = px.scatter_3d(irisDF,x="petal_width", y="sepal_length", z="petal_length", color="class", symbol="class", size="sepal_width",width=800)
fig.update_layout(legend=dict(x=0,y=1))
fig.show()
# %%
