#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hvplot.pandas

# %%
# Load the processed iris file
fileToLoad = "data/iris_clean.csv"
irisDF = pd.read_csv(fileToLoad)
irisDF.head()

# %%
# STEPS of Principal Components Analysis (PCA):
# Standarized data
# Initialize PCA model
# Apply Dimensionality reduction

#%%
# Standarize data with StandardScaler
irisScaled = StandardScaler().fit_transform(irisDF)
irisScaled[0:5]

# %%
# Initialize PCA model
pca = PCA(n_components=2)

#%%
# Get two principal components for the iris data.
irisPCA = pca.fit_transform(irisScaled)

# %%
# Transform PCA data to a dataframe
irisPCADF = pd.DataFrame(
    data=irisPCA, 
    columns=["Principal Component 1", "Principal Component 2"])
irisPCADF.head()

# %%
# Fetch the explained variance 
#   (to learn how much information can be attributed to each principal component)
pca.explained_variance_ratio_

# %%
# Find the best value for K
inertia = []
k = list(range(1, 11))

# Calculate the inertia for the range of K values
for i in k:
	km = KMeans(n_clusters=i, random_state=0)
	km.fit(irisPCADF)
	inertia.append(km.inertia_)

# Create the elbow curve
elbowData = {"k": k, "inertia": inertia}
elbowDF = pd.DataFrame(elbowData)
elbowDF.hvplot.line(x="k", y="inertia", xticks=k, title="Elbow Curve")


# %%
# Initialize the K-means model
model = KMeans(n_clusters=3, random_state=0)

# Fit the model
model.fit(irisPCADF)

# Predict clusters
predictions = model.predict(irisPCADF)

# Add the predicted class columns
irisPCADF["class"] = model.labels_
irisPCADF.head()

# %%
irisPCADF.hvplot.scatter(
	x="Principal Component 1",
	y="Principal Component 2",
	hover_cols=["class"],
	by="class",
)
# %%
