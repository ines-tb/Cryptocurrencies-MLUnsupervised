# %%
# Imports
# Preprocessing
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
# Reducing Data Dimensions
from sklearn.decomposition import PCA
# Clustering
from sklearn.cluster import KMeans
import hvplot.pandas
# Visualizing Results
import plotly.express as px


# ******************************************
#             DATA PREPROCESSING
# ******************************************

# %%
# Load file with data
fileToLoad: str = "Resources/crypto_data.csv"
cryptoRawDF: DataFrame = pd.read_csv(fileToLoad, index_col=0)
cryptoRawDF.head()

# %%
# Remove non-trading currencies
cryptoDF: DataFrame = cryptoRawDF[cryptoRawDF["IsTrading"]==True]

# %%
# Remove entries that do not have an Algorithm assigned
cryptoDF = cryptoDF[cryptoDF["Algorithm"]!=""]

# %%
# Drop IsTrading column
cryptoDF = cryptoDF.drop(["IsTrading"],axis=1)

# %%
# Remove null values
cryptoDF = cryptoDF.dropna()

# %%
# Remove currencies without Coins mined
cryptoDF = cryptoDF[cryptoDF["TotalCoinsMined"] > 0]

# %%
# Create a DataFrame with the coin names and same index as crypto dataframe
coinsNameDF: DataFrame = cryptoDF[["CoinName"]] 
coinsNameDF.set_index(cryptoDF.index, inplace=True)
#  Remove CoinName column
cryptoDF = cryptoDF.drop(["CoinName"],axis=1)

# %%
# Get dummy variables for the text columns
X: DataFrame = pd.get_dummies(
    cryptoDF, 
    columns=["Algorithm", "ProofType"])

# %%
# Standardize data with StandardScaler
XScaled: np.ndarray = StandardScaler().fit_transform(X)
XScaled[0:2]


# ******************************************
#     REDUCING DATA DIMENSIONS USING PCA
# ******************************************

#%%
# Initialize PCA model with 3 principal components
pca: PCA = PCA(n_components=3)

# %%
# Get two principal components for the iris data.
cryptoPca: np.ndarray = pca.fit_transform(XScaled)

# %%
# Transform PCA data to a dataframe
pcsDF: DataFrame = pd.DataFrame(
    data=cryptoPca,
    index=cryptoDF.index, 
    columns=["PC 1", "PC 2", "PC 3"])



# ******************************************
# CLUSTERING CRYPTOCURRENCIES USING K-MEANS
# ******************************************

# %%
# Calculate the inertia for the range of K values
inertia:list = []
k:list = list(range(1, 11))

for i in k:
   km: KMeans = KMeans(n_clusters=i, random_state=0)
   km.fit(pcsDF)
   inertia.append(km.inertia_)

# %%
# Create elbow curve plotting the inertia with hvplot
elbowData: dict = {"k": k, "inertia": inertia}
elbowDF: DataFrame = pd.DataFrame(elbowData)
elbowDF.hvplot.line(
    x="k", 
    y="inertia",
    xticks=k, 
    title="Elbow Curve")
# => As per the graph, best value K=4

# %%
# Run K-Means algorithm with k=4 as we found above to be the best value.
model: KMeans = KMeans(n_clusters=4, random_state=0)

# %%
# Fitting the model:
model.fit(pcsDF)

#%%
# Get Predictions:
predictions: np.ndarray = model.predict(pcsDF)
print(predictions)

# %%
# Add a new column to the iris DataFrame with the predicted classes
pcsDF['Class'] = model.labels_
pcsDF.head()

# %%
# Clustered DataFrame with all the information from K-Means model
clusteredDF: DataFrame = cryptoDF[["Algorithm", "ProofType", "TotalCoinsMined", "TotalCoinSupply"]]
clusteredDF.set_index(cryptoDF.index, inplace=True)
clusteredDF = clusteredDF.join(pcsDF)
clusteredDF = clusteredDF.join(coinsNameDF)

# %%
# Column order
columnOrder:list = ["Algorithm", "ProofType", "TotalCoinsMined", "TotalCoinSupply", "PC 1", "PC 2", "PC 3", "CoinName", "Class"]
clusteredDF = clusteredDF[columnOrder]



# ******************************************
#            VISUALIZING RESULTS
# ******************************************

# %%
# SCATTER 1:
# Plotting the clusters
fig = px.scatter_3d(
    clusteredDF,
    x="PC 1",
    y="PC 2",
    z="PC 3",
    color="Class",
    symbol="Class",
    # size="sepal_width",
    width=800,
    hover_name="CoinName",
    hover_data=["Algorithm"])

# %%
fig.update_layout(legend=dict(x=0,y=1))
fig.show()

# %%
# TABLE:
# Create hvplot table
clusteredDF.hvplot.table(
    columns=["CoinName", "Algorithm", "ProofType", "TotalCoinSupply", "TotalCoinsMined", "Class"], width=800)

# %%
# SCATTER 2:
clusteredDF.hvplot.scatter(
    x="TotalCoinsMined", 
    y="TotalCoinSupply", 
    hover_cols=["CoinName"],
    by="Class")

# %%
