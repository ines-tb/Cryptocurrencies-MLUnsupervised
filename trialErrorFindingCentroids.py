#%%
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import hvplot.pandas

#%%
# Load the file
filePath = "data/shopping_cleaned.csv"
shoppingDF = pd.read_csv(filePath)
shoppingDF.head()

# %%
# Plot two features
shoppingDF.hvplot.scatter(x="AnnualIncome", y="SpendingScore")

# %%
# Function to cluster and plot dataset
def test_cluster_amount(df, clusters):
    model = KMeans(n_clusters=clusters,random_state=5)
    model
    # Fitting the model
    model.fit(df)

    # Add a new class column
    df["class"] = model.labels_

# %%
# TWO clusters
test_cluster_amount(shoppingDF,2)
shoppingDF.hvplot.scatter(
    x="AnnualIncome",
    y="SpendingScore",
    by="class")
# => There are some points in the middel overlapped, let's see if it is because of the 2d dimension

# %%
fig = px.scatter_3d(
	shoppingDF,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
# THREE clusters
test_cluster_amount(shoppingDF,3)
fig = px.scatter_3d(
	shoppingDF,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
# FOUR CLUSTERS
test_cluster_amount(shoppingDF,4)
fig = px.scatter_3d(
	shoppingDF,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
# FIVE CLUSTERS
test_cluster_amount(shoppingDF,5)
fig = px.scatter_3d(
	shoppingDF,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# %%
# SIX CLUSTERS
test_cluster_amount(shoppingDF,6)
fig = px.scatter_3d(
	shoppingDF,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()


# %%
# SEVEN CLUSTERS
test_cluster_amount(shoppingDF,7)
fig = px.scatter_3d(
	shoppingDF,
    x="AnnualIncome",
	y="SpendingScore",
	z="Age",
    color="class",
	symbol="class",
	width=800,
)
fig.update_layout(legend=dict(x=0, y=1))
fig.show()

# => Increasing the number of cluster may refine the classes but as 
#   unsupervised learning does not have a defined outcome on what is being
#   measured, we could be adding complexity when adding clusters as we will
#   need to analyze the meaning and connection of those groups
# %%
