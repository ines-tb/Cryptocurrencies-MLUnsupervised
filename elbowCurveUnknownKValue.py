#%%
# Initial imports
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas

#%%
# Load data
filePath = "data/shopping_cleaned.csv"
shoppingDF = pd.read_csv(filePath)
shoppingDF.head(10)

# %%
# Inertia is the objective but K values is needed as well 
#   for calculation and plotting
inertia = []
k = list(range(1, 11))
# Calculate the inertia for the range of K values
for i in k:
   km = KMeans(n_clusters=i, random_state=0)
   km.fit(shoppingDF)
   inertia.append(km.inertia_)

# %%
# Create elbow curve plotting the inertia with hvplot
elbowData = {"k": k, "inertia": inertia}
elbowDF = pd.DataFrame(elbowData)
elbowDF.hvplot.line(x="k", y="inertia",xticks=k, title="Elbow Curve")
# => We can see that there are two angles in k=5 and k=6 that could be considered
#   k=3 is not considered because we look for a point where the vertical line shifts 
#    to a strong horizontal direction.

# %%
def get_clusters(k, data):
    # Create a copy of the DataFrame
    data = data.copy()
    # Initialize the K-Means model
    model = KMeans(n_clusters=k, random_state=0)
    # Fit the model
    model.fit(data)
    # Predict clusters
    predictions = model.predict(data)
    # Create return DataFrame with predicted clusters
    data["class"] = model.labels_
    return data

# %%
fiveClusters = get_clusters(5, shoppingDF)
fiveClusters.head()

# %%
sixClusters = get_clusters(6,shoppingDF)
sixClusters.head()

# %%
# Plotting the 2D-scatter with x="AnnualIncome" and y="SpendingScore"
fiveClusters.hvplot.scatter(x="AnnualIncome" , y="SpendingScore",by="class")

# %%
# Plot the 3D-scatter with x="Annual Income", y="Spending Score (1-100)" and z="Age"
fig = px.scatter_3d(
    fiveClusters,
    x="Age",
    y="SpendingScore",
    z="AnnualIncome",
    color="class",
    symbol="class",
    width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
# Plotting the 2D-scatter with x="AnnualIncome" and y="SpendingScore"
sixClusters.hvplot.scatter(x="AnnualIncome" , y="SpendingScore",by="class")

#%%
# Plotting the 3D-Scatter with x="Annual Income", y="Spending Score (1-100)" and z="Age"
fig = px.scatter_3d(
    sixClusters,
    x="Age",
    y="SpendingScore",
    z="AnnualIncome",
    color="class",
    symbol="class",
    width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# => Now the decision is up to the analyst, if we chose the 6 cluster
#       Cluster 0: medium income, low annual spend
#       Cluster 1: low income, low annual spend
#       Cluster 2: high income, low annual spend
#       Cluster 3: low income, high annual spend
#       Cluster 4: medium income, high annual spend
#       Cluster 5: very high income, high annual spend

# %%
