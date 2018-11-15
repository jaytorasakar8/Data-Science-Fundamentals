
# coding: utf-8

# In[1]:


import billboard
import spotipy
import pandas as pd


# In[2]:


chart = billboard.ChartData('hot-100', date='2018-11-10')


# In[3]:


dummychart = billboard.ChartData('hot-100', date='2018-11-10')


# In[6]:


chart1 = billboard.ChartData('hot-100', date='1999-01-01')
i =0
while chart1.previousDate:
    i = i+1
    chart1 = billboard.ChartData('hot-100', chart1.previousDate)
    #print(chart1.previousDate)
    if(chart1.previousDate <= '1998-09-30'):
        break;
print(i)
        


# In[9]:


def getData(date1, date2):
    chart = billboard.ChartData('hot-100', date1)
    list = []
    while chart.previousDate:
        date = chart.previousDate
        chart = billboard.ChartData('hot-100', chart.previousDate)
        i = 0;
        temptuples = ()
        
        while (i<100) :
            temptuples = (i, date, chart[i].title,chart[i].artist, chart[i].peakPos, chart[i].lastPos, chart[i].weeks, chart[i].rank, chart[i].isNew)
            i = i+1
            list.append(temptuples)
        if(chart.previousDate <= date2):
            break;
    
    return list

        
#print("Chart: ", date, "  ", chart[0].title, " ", chart[0].artist, " ", chart[0].peakPos, " ", chart[0].lastPos," ", chart[0].weeks," ", chart[0].rank," ", chart[0].isNew )


# In[10]:


finaldatalist1 = getData('2018-11-17', '2000-01-01')
#'1958-08-10'


# In[12]:


len(finaldatalist1)


# In[17]:


finaldatalist2 = getData('1999-12-31', '1985-01-01')


# In[19]:


len(finaldatalist2)


# In[25]:


#finaldatalist3_1 = getData('1984-12-31', '1980-01-01')


# In[26]:


#len(finaldatalist3_1)


# In[28]:


#finaldatalist3_2 = getData('1974-12-31', '1970-01-01')


# In[29]:


#finaldatalist3_3 = getData('1979-12-31', '1978-01-01')


# In[35]:


finaldatalist3_11 = getData('1984-12-31', '1978-01-01')


# In[36]:


finaldatalist3_22 = getData('1974-12-31', '1970-01-01')


# In[23]:


finaldatalist4 = getData('1969-12-31', '1958-08-10')


# In[13]:


df = pd.DataFrame(finaldatalist1, columns=['i', 'date', 'chart[i].title','chart[i].artist', 'chart[i].peakPos', 'chart[i].lastPos', 'chart[i].weeks', 'chart[i].rank', 'chart[i].isNew'])


# In[14]:


df.head(10)


# In[15]:


df.shape


# In[41]:


df_1 = pd.DataFrame(finaldatalist1, columns=['i', 'date', 'title','artist', 'peakPos', 'lastPos', 'weeks', 'rank', 'isNew'])
df_2 = pd.DataFrame(finaldatalist2, columns=['i', 'date', 'title','artist', 'peakPos', 'lastPos', 'weeks', 'rank', 'isNew'])
df_3 = pd.DataFrame(finaldatalist3_11, columns=['i', 'date', 'title','artist', 'peakPos', 'lastPos', 'weeks', 'rank', 'isNew'])
df_4 = pd.DataFrame(finaldatalist3_22, columns=['i', 'date', 'title','artist', 'peakPos', 'lastPos', 'weeks', 'rank', 'isNew'])
df_5 = pd.DataFrame(finaldatalist4, columns=['i', 'date', 'title','artist', 'peakPos', 'lastPos', 'weeks', 'rank', 'isNew'])


# In[42]:


data_set = pd.concat([df_5,df_4], axis=0)
print(data_set.shape)


# In[43]:


data_set = pd.concat([data_set,df_3], axis=0)
data_set = pd.concat([data_set,df_2], axis=0)
data_set = pd.concat([data_set,df_1], axis=0)
print(data_set.shape)


# In[45]:


data_set.tail(10)


# In[46]:


data_set.head()


# In[47]:


data_set.to_csv('bilboard_data.csv', sep=",")


# In[49]:


data_set = data_set.sort_values(by='date')
data_set.tail(10)


# In[50]:


data_set = data_set.reset_index()
data_set.tail(10)


# In[52]:


data_set.to_csv('bilboard_new_data.csv', sep=",")


# In[55]:


new_data = data_set.groupby(by='title').describe()


# In[63]:


new_data.columns.tolist()

