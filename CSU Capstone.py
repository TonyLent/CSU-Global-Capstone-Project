#!/usr/bin/env python
# coding: utf-8

# # **Telecom Churn Analysis**

# In[ ]:


# **Tony Lentini**


# ## **Problem Definition**
# 
#  - I see firsthand the importance and impact of churn in telecom. This has become an important area for telecom providers, as it costs more to attract new customers than to retain existing ones.
# 
# ### **The objective:**
# 
#  - Uncover common characteristics and correlations of customers who have already churned.
# 

# ------------------------------
# ## **Data Dictionary**
# ------------------------------
# 
# The dataset contains the following features:
# 
# 1. Customer: Unique Customer Identification.
# 2. Gender: The gender of the customer.
# 3. SeniorCitizen: Customer is 65 or older.
# 4. Partner: If the customer is married / partnered.
# 5. Dependents: If the customer lives with any dependents.
# 6. Tenure: The total amount of months that the customer has been with the company.
# 7. PhoneService: If the customer subscribes to home phone service with the company.
# 8. MultipleLines: If the customer subscribes to multiple telephone lines.
# 9. InternetService: If the customer subscribes to Internet service with the company.
# 10. OnlineSecurity: If the customer subscribes to an additional online security service provided by the company.
# 11. OnlineBackup: If the customer subscribes to an additional online backup service provided by the company.
# 12. DeviceProtection: If the customer subscribes to additional device protection plan for Internet equipment provided by the 
# company
# 13. TechSupport: If the customer subscribes to an additional technical support plan from the company.
# 143.StreamingTV: If the customer uses their Internet service to stream television programing from a third party provider.
# 15. StreamingMovies: If the customer uses their Internet service to stream movies from a third party provider.
# 16. Contract: The customer’s current contract type.
# 17. PaperlessBilling: If the customer has chosen paperless billing.
# 18. PaymentMethod: How the customer pays their bill.
# 19. MonthlyCharges: The customer’s current total monthly charge for all their services.
# 20. TotalCharges: The customer’s total charges, calculated to the end of the quarter.
# 21. Churn: If the customer left the company..

# ### **Loading Libraries**

# In[1]:


# Libraries to help with reading and manipulating data
import numpy as np
import pandas as pd

# Libraries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# To scale the data using z-score
from sklearn.preprocessing import StandardScaler

# To compute distances
from scipy.spatial.distance import cdist

# To perform K-means clustering and compute Silhouette scores
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# To visualize the elbow curve and Silhouette scores
from yellowbrick.cluster import SilhouetteVisualizer

# Importing PCA
from sklearn.decomposition import PCA

# To encode the variable
from sklearn.preprocessing import LabelEncoder

# Importing TSNE
from sklearn.manifold import TSNE

# To perform hierarchical clustering, compute cophenetic correlation, and create dendrograms
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet

# To compute distances
from scipy.spatial.distance import pdist

# To import K-Medoids
from sklearn_extra.cluster import KMedoids

# To import Gaussian Mixture
from sklearn.mixture import GaussianMixture

# To supress warnings
import warnings

warnings.filterwarnings("ignore")


# In[2]:


get_ipython().system('pip install yellowbrick')


# ### **Load the data**

# In[3]:


# loading the dataset
data = pd.read_csv("C:/Users/t_len/OneDrive/Documents/_Colorado State Univ/13 MIS_581/Telco_Customer_Churn2.csv")


# ### **Check the shape of the data**

# In[4]:


# Print the shape of the data
data.shape


# #### **Observations and Insights: **
# The shape tells us the dimensions of our dataset. It returns a tuple with the number of Rows and Columns in our dataset. 
# This is a good first step when working with a new dataset for analysis, as it tells us something about the volume of data
# we are dealing with. In this case we now know there are 21 variables (Columns) and 7043 Rows. 

# ### **Understand the data by observing a few rows**

# In[5]:


# View first 10 rows
# The head() method returns the first x rows (5 is the default if no number is specified) starting from the top of the dataset. 
# Given we have a significant amount of rows in our dataset, I will set this to return the first 25 rows. 
# This gives the Data Analyst a general idea of what the data looks like and the basic structure. 

data.head(10)


# In[12]:


# View last 10 rows Hint: Use tail() method
# The tail() method returns the last x rows (5 is the default if no number is specified) starting from the bottom of the dataset. 
# Given we have a significant amount of rows in our dataset, I will set this to return the first 25 rows. 

data.tail(10)


# ### **Check the data types and and missing values of each column** 

# In[7]:


# Check the datatypes of each column.
data.info()


# In[8]:


#Check the TotalCharges variable for uniqueness

print(data["TotalCharges"].unique())


# In[9]:


# Find the percentage of missing values in each column of the data
data.isnull().sum() / data.shape[0] * 100.00


# ## **Exploratory Data Analysis**

# ### **Explore the summary statistics of numerical variables**

# In[10]:


# Explore basic summary statistics of numeric variables. 
# "Summary Statistics" are descriptive statistics about the data in our dataset.
data[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].describe()


# In[11]:


data[["Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup"]].describe()


# In[12]:


data[["DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn"]].describe()


# ### **Explore the summary statistics of all categorical variables and the number of unique observations in each category**

# In[13]:


# List of the categorical columns in the data

cols = ["gender", "Dependents", "Partner", "PhoneService",  "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "Churn"]


# **Number of unique observations in each category**

# In[14]:


for column in cols:
    print("*" * 75)
    print("The Unique Values in the", column, "Variable are :")
    print(data[column].unique())
    
print("*" * 75)


# ## **2. Univariate analysis - Categorical Data**

# A function that will help create bar plots that indicate the percentage for each category. This function takes the categorical column as the input and returns the bar plot for the variable.

# In[15]:


#################################################################################
# Chi Squared Test (run once for each variable within the hypotheses) 
#################################################################################
import pandas as pd
from scipy.stats import chi2_contingency

df = pd.DataFrame(data)
f = pd.DataFrame(data)

# Create a contingency table (cross-tabulation)
contingency_table = pd.crosstab(df['Churn'], df['Contract'])

# Perform the Chi-Squared test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Squared Statistic: {chi2}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:\n", expected)

# Interpret the p-value
alpha = 0.05
if p_value < alpha:
    print("There is a significant association between Chrun and Contract (reject null hypothesis of independence).")
else:
    print("There is no significant association between Churn and Contract (fail to reject null hypothesis of independence).")


# ## **Bivariate Analysis**

# We have analyzed different categorical and numerical variables. Now, let's check how different variables are related to each other.

# ### **Correlation Heat map**
# Heat map can show a 2D correlation matrix between numerical features.

# In[21]:


plt.figure(figsize = (15, 7))
num_cols = data.select_dtypes(include = "number").columns.to_list()
sns.heatmap(data[num_cols].corr(), annot = True, vmin=-1, vmax=1, fmt=".2f", cmap="Spectral")

plt.show()

