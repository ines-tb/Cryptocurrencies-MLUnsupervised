#%%
# Initial imports
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas

#%%
# Loading data
filePath = "data/iris_clean.csv"
irisDF = pd.read_csv(filePath)
irisDF.head(10)

# %%
inertia = []
k = list(range(1, 11))

#%%
# Looking for the best K
for i in k:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(irisDF)
    inertia.append(km.inertia_)

# %%
# Define a DataFrame to plot the Elbow Curve using hvPlot
elbowData = {"k": k, "inertia": inertia}
elbowDF = pd.DataFrame(elbowData)
elbowDF.hvplot.line(x="k", y="inertia", title="Elbow Curve", xticks=k)
# => At point 3 there is the last angle of improvement with the shape of an elbow

# %%
