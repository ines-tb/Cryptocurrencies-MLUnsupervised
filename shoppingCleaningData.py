#%%
import pandas as pd
import re

#%%
fileToLoad = "Resources/shopping_data.csv"
shoppingDF = pd.read_csv(fileToLoad, encoding='ISO-8859-1')
shoppingDF.head()

# %%
# Columns
shoppingDF.columns

# %%
# All columns we use in unsupervised models must be numerical
# List DataFrame data types
shoppingDF.dtypes

# %%
# Unsupervised learning models can’t handle missing data
# Find null values
for i in shoppingDF.columns:
    print(f"Column {i} has {shoppingDF[i].isnull().sum()} null values")

# %%
# As the dataset has few null values we decide to handle them and not discard the dataset.
# Questions to decide:
# Are there string columns that we can’t use? 
# Are there columns with excessive null data points? 
# Was our decision to handle missing values or just remove them?

# Drop null rows
shoppingDF = shoppingDF.dropna()

# %%
# Find duplicate entries
print (f"Duplicate entries: {shoppingDF.duplicated().sum()}")

# %%
# Remove CustomerID as it does not tell anything
shoppingDF.drop(columns=["CustomerID"], inplace=True)
shoppingDF.head()

# %%
# DATA IS SET UP FOR UNSUPERVISED:
# 1. Null values are handled.
# 2. Only numerical data is used.
# 3. Values are scaled. In other words, data has been manipulated to ensure that the
#    variance between the numbers won’t skew results.

# Transform 'Card Member' column to translate strings to numbers
def change_string(member):
    if member == 'Yes':
        return 1
    else:
        return 0

shoppingDF["Card Member"] = shoppingDF["Card Member"].apply(change_string)
shoppingDF.head()

# %%
# Rescale Annual Income to match the rest of the columns
shoppingDF["Annual Income"] = shoppingDF["Annual Income"] / 1000
shoppingDF.head()

# %%
# Reformat the names of the columns so they contain no spaces or numbers.
def reformat_columns_names(name: str) -> str:
    newName = re.sub("\((\d{1}-\d{3})\)","",name).replace(" ","")
    return newName 
    
for i in shoppingDF.columns:
    shoppingDF[reformat_columns_names(i)] = shoppingDF[i]

shoppingDF.drop(columns=["Card Member","Annual Income", "Spending Score (1-100)"],inplace=True)

shoppingDF.head()

# %%
# Save cleaned data
outputFilePath = "data/shopping_cleaned.csv"
shoppingDF.to_csv(outputFilePath,index=False)

# %%
# Save cleaned data to excel;
outputFilePathExcel = "data/shopping_cleaned.xlsx"
shoppingDF.to_excel(outputFilePathExcel,index=False)

# %%
# Save cleaned data to json;
outputFilePathJson = "data/shopping_cleaned.json"
shoppingDF.to_json(outputFilePathJson)

# %%
