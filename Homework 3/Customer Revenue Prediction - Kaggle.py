
# coding: utf-8

# # Google Customer Revenue Prediction 
# 
# We aim to predict how much Google Store customers will spend on the Google products 

# #### Data Import and Exploration

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import seaborn as sb
import time
import json
import datetime as dt

from sklearn import metrics
from pandas.io.json import json_normalize
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train_df =  pd.read_csv('./train3.csv') 
train_df.dtypes


# In[3]:


print(train_df.shape[0])


# ##### We see that our dataset is of 900K tuples

# In[4]:


train_df.head()


# #### Data is in JSON Format for the columns: Device, geonetworks, totals and traffic source
# Since the data is in JSON Format we convert the given data into standard format

# In[5]:


columns_in_json = ['device', 'geoNetwork', 'totals', 'trafficSource']

#We need to reload the Dataframe with all the data formatting 

def load_df(path_name):
    df = pd.read_csv(path_name, converters = {column: json.loads for column in columns_in_json}, dtype = {'fullVisitorId' : 'str'}, nrows = None )

    for column in columns_in_json:
        json_column_as_df = json_normalize(df[column])
        json_column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in json_column_as_df.columns]
        df = df.drop(column, axis = 1).merge(json_column_as_df, right_index = True, left_index = True)
    
    return df

#Reference: https://medium.com/@gis10kwo/converting-nested-json-data-to-csv-using-python-pandas-dc6eddc69175    


# In[6]:


train_df = load_df('./train3.csv')
test_df = load_df('./test3.csv')


# In[7]:


pd.set_option('display.max_columns', None)
test_df.head()


# In[8]:


pd.set_option('display.max_columns', None)
train_df.head()

#For displaying all columns 
#Source: https://stackoverflow.com/questions/28775813/not-able-to-view-all-columns-in-pandas-data-frame 


# #### Format the given date in regular format of Y/M/D format from the given POSIX format

# In[9]:


temp = train_df['date'].apply(lambda x: dt.datetime.strptime(str(x), "%Y%m%d") )
train_df['date'] = temp

#Reference: https://stackoverflow.com/questions/30132282/datetime-to-string-with-series-in-python-pandas


# #### We are calculating the Log value of the Transaction Revenue! 

# In[10]:


log_values = train_df['totals.transactionRevenue'].fillna(0).astype(float)
log_values = log_values.apply(lambda x: np.log1p(x))
train_df['totals.transactionRevenue'] = log_values
train_df['totals.transactionRevenue'].describe()


# #### We are trying to find the columns which are constant and not having any impact on our prediction  
# 

# In[11]:


const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]
const_cols


# In[12]:


print("Number of unique visitors in train set : ",train_df.fullVisitorId.nunique())


# In[13]:


print("Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns)))


# #### We need to drop columns which are not present in both the train and test data set

# In[14]:


cols_to_drop = const_cols + ['sessionId']

train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)


# In[15]:


train_df.columns.values


# #### We are looking in Channel Grouping Distribution Statistics  

# In[16]:


channel_group_column = train_df["channelGrouping"].unique()
channel_group_count = train_df["channelGrouping"].value_counts()

plt.figure(figsize = (10,8))
plt.bar(channel_group_column, channel_group_count, align='center', alpha=1)
plt.ylabel('Count')
plt.title('Channel Grouping Distribution Stats')
plt.show()


# ### Plots based on the Devices and it's various Categories

# In[17]:


fig, axes = plt.subplots(2,2,figsize=(20,20))

train_df["device.browser"].value_counts().head(6).plot.bar(ax=axes[0][0],rot=30, title="Browser")
train_df["device.deviceCategory"].value_counts().plot.bar(ax=axes[0][1],rot=0,title="Category of Device")
train_df["device.isMobile"].value_counts().plot.bar(ax=axes[1][0],rot=0,title="Mobile Usage")
train_df["device.operatingSystem"].value_counts().head(7).plot.bar(ax=axes[1][1],rot=30,title="Operating System")

#Reference: https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.plot.bar.html


# In[18]:


fig, axes = plt.subplots(3,2,figsize=(20,20))

train_df["geoNetwork.continent"].value_counts().plot.bar(ax=axes[0][0],rot=30, title="Continent")
train_df[train_df["geoNetwork.continent"] == "Americas"]["geoNetwork.subContinent"].value_counts().plot.bar(ax=axes[0][1], rot=0,title="American Continent Data")
train_df[train_df["geoNetwork.continent"] == "Asia"]["geoNetwork.subContinent"].value_counts().plot.bar(ax=axes[1][0],rot=0,title="Asian Continent Data")
train_df[train_df["geoNetwork.continent"] == "Europe"]["geoNetwork.subContinent"].value_counts().plot.bar(ax=axes[1][1],rot=0,title="European Continent Data")
train_df[train_df["geoNetwork.continent"] == "Africa"]["geoNetwork.subContinent"].value_counts().plot.bar(ax=axes[2][0],rot=0,title="African Continent Data")
train_df[train_df["geoNetwork.continent"] == "Oceania"]["geoNetwork.subContinent"].value_counts().plot.bar(ax=axes[2][1],rot=0,title="Oceanian Continent Data")

#Reference: https://stackoverflow.com/questions/29498652/plot-bar-graph-from-pandas-dataframe


# In[19]:


plt.figure(figsize=(16,8))
train_df["geoNetwork.continent"].value_counts().plot.pie(label='Continent-Wise Distribution')
plt.axis('equal')


# In[52]:


plt.figure(figsize = (8,6))
plt.xlabel('Page Views', fontsize = 14)
plt.ylabel('Hits', fontsize = 14)
plt.scatter(train_df["totals.pageviews"], train_df["totals.hits"])
plt.show()


# In[55]:


plt.figure(figsize = (8,6))
plt.xlabel('City', fontsize = 14)
plt.ylabel('Metro', fontsize = 14)
plt.scatter(train_df["geoNetwork.city"], train_df["geoNetwork.metro"])
plt.show()


# In[20]:


revenue_datetime_df = train_df[["totals.transactionRevenue" , "date"]].dropna()
revenue_datetime_df["revenue"] = revenue_datetime_df["totals.transactionRevenue"].astype(np.int64)
revenue_datetime_df.head()


# In[21]:


daily_revenue_df = revenue_datetime_df.groupby(by=["date"],axis = 0 ).sum()

fig, axes = plt.subplots(figsize=(16,8))

axes.set_title("Revenue Generated based on Date")
axes.set_xlabel("Date")
axes.set_ylabel("Revenue Generated")
axes.plot(daily_revenue_df)


# ### Data Processing based on Category and Numerical Variables

# In[22]:


#Splitting the categorical variables and Numerical Variables

from sklearn import preprocessing
categorical_columns = ['channelGrouping', 'device.browser',
       'device.deviceCategory','device.operatingSystem', 'geoNetwork.city',
       'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro',
       'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent', 'trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
       'trafficSource.isTrueDirect', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source']

for column in categorical_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[column].values.astype('str')) + list(test_df[column].values.astype('str')))
    train_df[column] = lbl.transform(list(train_df[column].values.astype('str')))
    test_df[column] = lbl.transform(list(test_df[column].values.astype('str')))


# In[23]:


#For columns with numerical values
numerical_columns = ['totals.bounces', 'totals.hits',
       'totals.newVisits', 'totals.pageviews',
       'visitNumber', 'visitStartTime']

for column in numerical_columns:
    train_df[column] = train_df[column].astype(float)
    test_df[column] = test_df[column].astype(float)


# In[24]:


train_df.describe()


# ### Coorelation and HeatMap
# We are generating heatmap by taking the corelation between multiple parameters provided to us

# In[25]:


#Generating Coorelation And HeatMap

correlation_df = train_df[[i for i in list(train_df.columns) if i not in ['totals.bounces', 'totals.newVisits']]]
corr = correlation_df.corr()
fig, ax = plt.subplots(figsize=(15, 15))
sb.heatmap(corr,  linewidths=.5, annot=False)

#Reference: https://seaborn.pydata.org/generated/seaborn.heatmap.html


# In[26]:


correlation_df1 = train_df[[i for i in list(train_df.columns) if i in ['visitId', 'visitStartTime','device.deviceCategory','device.isMobile', 'totals.hits','totals.pageviews', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot'] ]];
corr1 = correlation_df1.corr()
fig, ax = plt.subplots(figsize=(8, 8))
sb.heatmap(corr1,  linewidths=.5, cmap="YlGnBu", annot=False)


# * The above heatmap is generated based on a few selected features which are having high coorelation value amongst the features

# ### External Data Set 
# 
# I have taken data from the source: https://www.kaggle.com/satian/exported-google-analytics-data. We are trying to see if we can get meaningful datasets from the given dataset. Using the external dataset, we can enhance our prediction model and we can improve our results which will help us improve the efficiency of our result 

# In[27]:


train_copy_df = train_df
test_copy_df = test_df


# In[28]:


external_train_df =  pd.read_csv('./ExternalData/Train_external_data.csv', dtype = {"Client Id": 'str'}) 
external_train_df.head()


# In[29]:


external_test_df =  pd.read_csv('./ExternalData/Test_external_data.csv', dtype = {"Client Id": 'str'}) 
external_test_df.head()


# ##### Data Cleaning of the External Data set and merging them

# In[30]:


for df in [external_train_df,external_test_df]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[0]).astype(str)

categorical_columns = ['Revenue', 'Sessions', 'Avg. Session Duration', 'Bounce Rate', 'Transactions', 'Goal Conversion Rate', 'visitId']

for column in categorical_columns:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(external_train_df[column].values.astype('str')) + list(external_test_df[column].values.astype('str')))
    external_train_df[column] = lbl.transform(list(external_train_df[column].values.astype('str')))
    external_test_df[column] = lbl.transform(list(external_test_df[column].values.astype('str')))

external_train_df = external_train_df.merge(external_train_df, how="left", on="visitId")
external_test_df = external_test_df.merge(external_test_df, how="left", on="visitId")

for df in [external_train_df,external_test_df]:
    df.drop("Client Id", axis=1, inplace=True)


# ## Baseline Model - Light GDM model 

# In[31]:


import lightgbm as lgb
params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }


# In[32]:


#Remove transactionRev from train_df. That will be train_y
final_train_y = train_df["totals.transactionRevenue"]

del train_df["totals.transactionRevenue"]
final_train_df = train_df


# In[33]:


val_X = final_train_df[categorical_columns + numerical_columns] 
val_y = final_train_y


# In[34]:


test_X = test_df[categorical_columns + numerical_columns]


# In[35]:


date_data = final_train_df["date"]
fullVisitorId_data = final_train_df["fullVisitorId"]


# In[36]:


del final_train_df["date"]
del final_train_df["fullVisitorId"]


# In[37]:


final_train_df.dtypes


# In[38]:


test_X.dtypes


# In[39]:


print("Difference:", set(final_train_df.columns).difference(set(test_X.columns)))
del final_train_df["visitId"]
del final_train_df["device.isMobile"]


# In[40]:


def data_train(train_X, train_y, val_X, val_y, test_X):   
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

pred_test, model, pred_val = data_train(final_train_df, final_train_y, val_X, val_y, test_X)

#Reference: https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc


# In[42]:


#Finding the Mean Squared Error

pred_val[pred_val<0] = 0

val_pred_df = pd.DataFrame({"fullVisitorId":fullVisitorId_data.values})
val_pred_df["transactionRevenue"] = final_train_y.values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()

print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


# In[43]:


test_id = test_df["fullVisitorId"].values


# In[44]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0

sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]

sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("result.csv", index=False)


# #### Predicting of Buying with Probablility Funciton using Logistic Regression

# In[154]:


from sklearn.linear_model import LogisticRegression

logistic_df = final_train_df 
logistic_columns = ['totals.hits','totals.newVisits', 'totals.pageviews', 'totals.bounces']
logistic_df[logistic_columns] = logistic_df[logistic_columns].fillna(0.0).astype(int)
#logistic_df.dtypes

temp = final_train_y > 0.0 
temp = temp.astype(int)
new_df = pd.DataFrame() 
new_df['target_value'] = temp


feature_vars = numerical_columns + categorical_columns 
log_features = logistic_df[feature_vars].drop(['geoNetwork.region','trafficSource.adwordsClickInfo.gclId',
                                       'trafficSource.campaign','trafficSource.keyword','trafficSource.adwordsClickInfo.slot',
                                       'trafficSource.adwordsClickInfo.gclId'], axis=1)

feature_vars_1 = ['totals.bounces', 'totals.hits', 'totals.newVisits',
       'totals.pageviews', 'visitNumber', 'visitStartTime',
       'channelGrouping', 'device.browser', 'device.deviceCategory',
       'device.operatingSystem', 'geoNetwork.city',
       'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro',
       'geoNetwork.networkDomain', 'geoNetwork.subContinent',
       'trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.isTrueDirect', 'trafficSource.medium',
       'trafficSource.referralPath', 'trafficSource.source']

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(logistic_df[feature_vars_1], new_df) 
clf.predict_proba(logistic_df[feature_vars_1])[:,0]


# In[155]:


regression_df = train_df
regression_features = list(regression_df.columns.values)

regression_df = regression_df[regression_features].drop(['geoNetwork.region','trafficSource.adwordsClickInfo.gclId','trafficSource.adwordsClickInfo.slot','trafficSource.campaign','trafficSource.keyword'], axis=1)


# In[156]:


regression_df['probs'] = clf.predict_proba(log_features)[:,0]
regression_df = regression_df.sort_values(by='probs', ascending=False)


# In[158]:


regression_df.head(10)


# ### Using another Model - Random Forest for prediction

# In[159]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

permutation_df = final_train_df 

train_x, test_x, train_y, test_y = train_test_split(permutation_df, final_train_y, test_size=0.2)
rf = RandomForestRegressor(n_estimators = 10)
model = rf.fit(train_x, train_y)


# In[169]:


y_pred = model.predict(train_x)


# * We don't see any improvement in the results as compared to LGDM model, so didn't advance further

# #### Permutation Test p-values
# We are doing the Permutation Test, and it is done inorder to see the effects of the data shuffle on the final RMSE value 

# In[160]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf).fit(test_x, test_y)
eli5.show_weights(perm, feature_names = test_x.columns.tolist())

#Reference: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html


# *From above we can see that the feature of pageviews is the most important one in this given dataset*
