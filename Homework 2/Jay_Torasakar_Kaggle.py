
# coding: utf-8

# # New York City Taxi Fare Prediction
# 
# The aim is to predict the taxi fare for the customer taking cab service in New York City

# #### Data Import and Exploration

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


print(os.listdir('./'))


# In[3]:


train_df =  pd.read_csv('./train.csv') 
train_df.dtypes


# In[4]:


print(train_df.shape[0])


# In[5]:


#Drop any values that are NaN
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))


# In[6]:


#Drop all the rows that have value = 0 in them
train_df = train_df[(train_df != 0).all(1)] 
print('New size: %d' % len(train_df))


# In[7]:


#Take only those Fares of passengers that have a postive value
train_df = train_df[train_df.fare_amount > 0]
print('New size: %d' % len(train_df))


# In[8]:


#Using the general fare_amount from the given data set, it seems the fare of more than 175, are outliers
train_df = train_df[train_df.fare_amount < 175]
print('New size: %d' % len(train_df))


# In[9]:


#We are taking percentile of the data in order to identify the outliers and make sure to remove them 
train_df['pickup_longitude'].quantile([0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95, 0.99, 1])


# * From the above values we can see that the majority of data lies between 1%tile and 99%tile. So we are now going to filter the data according to their percentile ranges   

# In[10]:


print('Size of training data: %d' % len(train_df) )

train_df = train_df[(train_df['pickup_longitude'] <= train_df['pickup_longitude'].quantile(0.999)) ]
train_df = train_df[(train_df['pickup_longitude'] >= train_df['pickup_longitude'].quantile(0.005))]

train_df = train_df[(train_df['pickup_latitude'] <= train_df['pickup_latitude'].quantile(0.999)) ]
train_df = train_df[(train_df['pickup_latitude'] >= train_df['pickup_latitude'].quantile(0.005))]

train_df = train_df[(train_df['dropoff_longitude'] <= train_df['dropoff_longitude'].quantile(0.999)) ]
train_df = train_df[(train_df['dropoff_longitude'] >= train_df['dropoff_longitude'].quantile(0.005))]

train_df = train_df[(train_df['dropoff_latitude'] <= train_df['dropoff_latitude'].quantile(0.999)) ]
train_df = train_df[(train_df['dropoff_latitude'] >= train_df['dropoff_latitude'].quantile(0.005))]

print('New size: %d' % len(train_df))


# ##### We tried taking different values of the percentile, and taking the data between 0.5%tile and 99.9%tile gives us the best data set for training. 

# In[11]:


eu_cal = (train_df['dropoff_latitude'] - train_df['pickup_latitude']) **2  + (train_df['dropoff_longitude'] - train_df['pickup_longitude']) **2

eu_dist = np.sqrt(eu_cal)


# ### Pearson Correlation between Eucledian Distance and Fare Amount

# In[12]:


eu_dist.corr(train_df['fare_amount'])


# In[13]:


#Get the given pickup time in a new format of date of Y-M-D:H-M-S
train_df['pickup_datetime'] = train_df['pickup_datetime'].str.replace(" UTC", "")
#replace the given date time in a new format
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


# In[14]:


journey_time = (train_df['pickup_datetime'].dt.hour)*60 + train_df['pickup_datetime'].dt.minute


# ### Pearson Correlation between Eucledian Distance and the Time

# In[15]:


eu_dist.corr(journey_time)


# ### Pearson Correlation between Fare Amount and the Time

# In[16]:


journey_time.corr(train_df['fare_amount'])


# In[17]:


train_df['distance'] = eu_dist
train_df['journey_time'] = journey_time


# ### Plot between Distance and Total Fare Amount

# In[18]:


plt.figure(figsize = (8,6))
plt.xlabel('Distance Traveled', fontsize = 14)
plt.ylabel('Total Fare Amount', fontsize = 14)
plt.scatter(train_df[:1200].distance, train_df[:1200].fare_amount)
plt.show()


# * The plot between Distance traveled and the Total Fare Amount generates a linear relationship

# ### Plot between Distance and Time

# In[19]:


plt.figure(figsize = (8,6))
plt.xlabel('Distance Traveled', fontsize = 14)
plt.ylabel('Total Journey Time (in minutes)', fontsize = 14)
plt.scatter(train_df[:1200].distance, train_df[:1200].journey_time)
plt.show()


# * The plot between Distance traveled and the total journey time of a passenger generates a non - linear relationship. This plot doesn't tell us the exact relationship between the variables

# ### Plot between Total Fare Amount and Time

# In[20]:


plt.figure(figsize = (8,6))
plt.xlabel('Total Journey Time (in minutes)', fontsize = 14)
plt.ylabel('Total Fare Amount', fontsize = 14)
plt.scatter(train_df[:1200].journey_time, train_df[:1200].fare_amount)
plt.show()


# * The plot between Distance traveled and the total journey time of a passenger generates a non - linear relationship. The plot gives a scattered data and hence we cannot infer anything from it

# ### Plot between Total Fare Amount and Total Passenger Count

# In[22]:


plt.figure(figsize = (12,6))
plt.xlabel('Total Fare Amount', fontsize = 14)
plt.ylabel('Total Passengers Count', fontsize = 14)
plt.scatter(train_df[:1200].fare_amount, train_df[:1200].passenger_count)
plt.show()


# * The plot between total fare and the number of passengers taking the cab, we see that generally people in New York City are spending less than 30$ while they take the cab service, although it is a very generic statement

# ### Bar Chart comparing the fare amount depending on the hour of the day

# In[23]:


pickup_hour_time = (train_df['pickup_datetime'].dt.hour)[:1200]
fare = train_df[:1200].fare_amount

plt.figure(figsize = (10,6))
plt.bar(pickup_hour_time, fare)
plt.xlabel('Hour of the day')
plt.ylabel('Fare amount')
plt.show()


# * This plot shows relationship between the fare amount at particular hours of the day. We can see at what times during the day, we have highest amount of fare. It seems that during early morning hours(3 AM, 9 AM), and the evening at 5PM, 6PM and 8 PM we are able to see the highest fare amount throughout the day. 

# ### Data plot based on Pickup locations 

# In[24]:


plt.figure(figsize = (12,8))
plt.title('Pickup Locations', fontsize=14)
p_long, p_lat = pd.Series(train_df[:1200].pickup_longitude, name="Pickup Longitude"), pd.Series(train_df[:1200].pickup_latitude, name="Pickup Latitude")
pickup = sb.regplot(x=p_long, y=p_lat, marker="+")


# * From the given plot we get to know the various locations of pickups based on their latitude and longitude coordinates. Since we have a huge datasset, we can see the whole map of New York City from the marking locations of pickup. We can also see there are a few points for the pickup in the right bottom corner, because those coordinates are of JFK airport, while a few scattered points are also present which represent the pickup locations in the New York's boroughs as well. So we can take the pickup based on these locations as well.

# ### Data plot based on DropOff locations 

# In[25]:


plt.figure(figsize = (12,8))
plt.title('Dropoff Locations', fontsize=14)
d_long, d_lat = pd.Series(train_df[:1200].dropoff_longitude, name="Dropoff Longitude"), pd.Series(train_df[:1200].dropoff_latitude, name="Dropoff Latitude")
dropoff = sb.regplot(x=d_long, y=d_lat, marker="+")


# * From the given plot we get to know the various locations of dropoff points based on their latitude and longitude coordinates. We can see the map of New York City from the marking locations of dropoff. We can also see there are a few points for the pickup in the right bottom corner, because those coordinates are of JFK airport. Here we see a lot of scattered points across Manhattan, so we can say that, many of the taxi cab's customers have a drop off location outside of New York City.

# ### Addtional Feature Extraction

# In[26]:


train_df['day'] = train_df['pickup_datetime'].dt.day;
train_df['month'] = train_df['pickup_datetime'].dt.month;
train_df['hour'] = train_df['pickup_datetime'].dt.hour;
train_df['minute'] = train_df['pickup_datetime'].dt.minute;
train_df.head()


# ### Data Training - Linear Regression

# In[27]:


data2 = train_df[:5000000]

features = ['passenger_count', 'distance','journey_time']
X = data2[features]
y = data2['fare_amount']

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)


# In[28]:


print(lr.intercept_)
print(lr.coef_)
zip(features, lr.coef_)


# In[29]:


y_pred = lr.predict(X_test)
print(y_pred)
print("RMS: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# #### We are getting RMS value =  3.912 from the training data set 

# ### Actual Data Prediction using Training Data Set

# In[30]:


actual_data =  pd.read_csv('./test.csv') 
actual_data.dtypes


# In[31]:


eu_cal = (actual_data['dropoff_latitude'] - actual_data['pickup_latitude']) **2  + (actual_data['dropoff_longitude'] - actual_data['pickup_longitude']) **2

eu_dist = np.sqrt(eu_cal)


# In[32]:


actual_data['pickup_datetime'] = actual_data['pickup_datetime'].str.replace(" UTC", "")
#replace the given date time in a new format
actual_data['pickup_datetime'] = pd.to_datetime(actual_data['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


# In[33]:


journey_time = (actual_data['pickup_datetime'].dt.hour)*60 + actual_data['pickup_datetime'].dt.minute


# In[34]:


actual_data['distance'] = eu_dist
actual_data['journey_time'] = journey_time


# In[35]:


actual_data['day'] = actual_data['pickup_datetime'].dt.day;
actual_data['month'] = actual_data['pickup_datetime'].dt.month;
actual_data['hour'] = actual_data['pickup_datetime'].dt.hour;
actual_data['minute'] = actual_data['pickup_datetime'].dt.minute;
actual_data.head()


# In[36]:


features1 = ['passenger_count', 'distance','journey_time']
X1 = actual_data[features1]


# In[37]:


predict_value = lr.predict(X1)
print(predict_value)


# In[38]:


final_data = pd.DataFrame()
final_data['key'] = actual_data['key']
final_data['fare_amount'] = predict_value
final_data.to_csv('final_result.csv',sep=',', index = False)

