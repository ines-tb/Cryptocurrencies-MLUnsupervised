#%%
import pandas as pd

#%%
irisDF = pd.read_csv("Resources/iris.csv")
irisDF.head()

# %%
# Unsupervised ML deals with numerical data, so either drop or convert any non-numerical
newIrisDF = irisDF.drop(['class'], axis=1)

# %%
newIrisDF = newIrisDF[['sepal_length','petal_length','sepal_width','petal_width']]

# %%
outputFilePath = 'data/iris_clean.csv'
newIrisDF.to_csv(outputFilePath,index=False)

# %%
