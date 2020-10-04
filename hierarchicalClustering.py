#%%
# Imports
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import hvplot.pandas
import plotly.figure_factory as ff



#%%
# Load the processed iris file
fileToLoad = "data/iris_clean.csv"
irisDF = pd.read_csv(fileToLoad)
irisDF.head()

# %%
# Apply PCA to reduce the dataset from four features to two

# Standarize data with StandardScaler
irisScaled = StandardScaler().fit_transform(irisDF)
# Initialize PCA model
pca = PCA(n_components=2)
# Get two principal components for the iris data.
irisPCA = pca.fit_transform(irisScaled)
# Transform PCA data to a dataframe
irisPCADF = pd.DataFrame(
    data=irisPCA, 
    columns=["Principal Component 1", "Principal Component 2"])
# Fetch the explained variance 
#   (to learn how much information can be attributed to each principal component)
print(pca.explained_variance_ratio_)


# %%
# Create the dendrogram
# -------------------------
fig = ff.create_dendrogram(irisPCADF, color_threshold=0)
fig.update_layout(width=800, height=500)
fig.show()
# => We know the iris dataset contains three clusters. The cutoff will be set at five to obtain three clusters
#    We knew ahead of time the number of clusters to make; however, the cutoff line on the dendrogram seems 
#    high in terms of distances. This is one of the difficulties when using a dendrogram


# %%
# Hierarchical clustering
# ------------------------------
# agg = AgglomerativeClustering(n_clusters=7)
agg = AgglomerativeClustering(n_clusters=3)
model = agg.fit(irisPCADF)

# %%
# Add a new class column to irisPCADF
irisPCADF["class"] = model.labels_
irisPCADF.head()

# %%
# Plot to show the results of the hierarchical clustering algorithm
irisPCADF.hvplot.scatter(
    x="Principal Component 1",
    y="Principal Component 2",
    hover_cols=["class"],
    by="class",
)
# %%
