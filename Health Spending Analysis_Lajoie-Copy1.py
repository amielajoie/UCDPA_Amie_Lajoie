#!/usr/bin/env python
# coding: utf-8

# # Introduction: Global Health Spending
# 
# Data Source: https://www.kaggle.com/datasets/danevans/world-bank-wdi-212-health-systems
# 
# In this analysis, I explore the spending by countries around the world in terms of health systems and national health expenditure (the main data set used for this analysis is from 2016). I also scrutinise a number of relationships as portrayed by the data: for example between the spending as a percentage of a country's GDP and the number of Physicians in a country per 1000 persons. We also look more closely at data from the top 10 countries who spend the most per capita and per percentage of overall GDP on health costs. To conclude, we add an API to continue the analysis at another time! 
# 

# In[1]:


import pandas as pd
import os
working_directory = os.getcwd()
print(working_directory)


# In[2]:


path = working_directory + '/downloads/2.12_Health_systems.csv'
df = pd.read_csv(path)

df.head(5)


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[19]:


#Replacing NaN values to 0, Dropping duplicates
df_clean = df.fillna(0)
df.drop_duplicates(inplace = True)


# In[ ]:


#Dropping columns
del df_clean['Province_State']


# In[20]:


df_clean.info()


# In[27]:


#Sorting Health spending out of pocket (high to low)

df_clean.sort_values(by = ['Health_exp_out_of_pocket_pct_2016'], ascending = False )


# In[29]:


#Which 10 countries spent the most per capita (in USD) on health?

df_clean.nlargest(10, 'Health_exp_per_capita_USD_2016')


# In[36]:


#Looking only at data from Ireland

df_clean.loc[88]


# In[22]:


#Using NumPy to find the median & mean of global health spending per capita (no statistical difference)

gdp = df_clean['Health_exp_pct_GDP_2016']
np.gdp = np.array(gdp)
gdp_mean = np.mean(np.gdp)
gdp_median = np.median(np.gdp)

print(gdp_mean)
print(gdp_median)


# In[25]:


#Finding countries with above mean/median spending of GDP on health 

above_avg_gdp = df_clean[(df_clean['Health_exp_pct_GDP_2016']>= 5.95)]

#Finding countries with above mean/median spending of GDP on health and with 5 or more Physicians per 1000 person

high_gdp_physicans = df_clean[(df_clean['Health_exp_pct_GDP_2016']>= 5.95) & (df_clean['Physicians_per_1000_2009-18'] >= 5)]


# In[24]:


#Looping a list: Countries with above average % of GDP spending & with more than 5 physicians per 1000 people

physicans = [['Austria', 5.1], ['Cuba', 8.2], ['Georgia', 5.1], ['San Marino', 6.1], ['Sweden', 5.4], ['Uruguay', 5.0]]
for x, y in physicans : 
    print('number of physicans in ' + x + 'is ' + str(y) + ' per 1000 people')


# In[40]:


#Relationship between spending as % of a country's GDP, and the number of physicians per 1000 people?

plt.scatter(df_clean['Health_exp_pct_GDP_2016'], df_clean['Physicians_per_1000_2009-18'])
plt.xlabel('Health Spending as Percentage of GDP')
plt.ylabel('Number of Physicians per 1000 persons')
plt.title('Relationship between Health Spending as % of GDP and the Number of Physicians')


# In[129]:


#Relationship between spending as % of a country's GDP, and the number of nurses per 1000 people?

plt.scatter(df_clean['Health_exp_pct_GDP_2016'], df_clean['Nurse_midwife_per_1000_2009-18'])
plt.xlabel('Health Spending as Percentage of GDP')
plt.ylabel('Nurse_Midwives_per 1000 people')


# In[73]:


#Creating DataFrame by Index for Top 10

top10 = {'Label': [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
         'Country': [ 'United States', 'Switzerland', 'Norway', 'Luxembourg', 'Sweden', 'Denmark', 'Iceland', 'Australia', 'Ireland', 'The Netherlands'],
         '2016 Health spending % gdp': [17.1, 12.2, 10.5, 6.2, 10.9, 10.4, 8.3, 9.3, 7.4, 10.4],
         '2016 Health spending per capita': [9869.7, 9836.0, 7477.9, 6271.4, 5710.6, 5565.6, 5063.6, 5002.4, 4758.6, 4742.0]}

top10_df = pd.DataFrame(top10).set_index('Label')
top10_df


# In[78]:


plt.figure(figsize = (16, 5))
sns.barplot(x = top10_df['Country'], y = top10_df['2016 Health spending % gdp'])
plt.title('Health Spending as % GDP for Countries who spend the most on Health per Capita')

plt.show


# In[130]:


#New DataFrame: Adding 2019 figures for GDP spending on healthcare

r = requests.get('https://www.statista.com/statistics/268826/health-expenditure-as-gdp-percentage-in-oecd-countries/')
type(r)

newtop10 = {'Label': [ '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'Country': [ 'United States', 'Switzerland', 'Norway', 'Luxembourg', 'Sweden', 'Denmark', 'Iceland', 'Australia', 'Ireland', 'The Netherlands'],
        '2019 Health spending % gdp': [ 16.8, 11.3, 10.5, 6.2, 10.9, 10, 8.6, 9.4, 6.7, 10.2]}

newtop10_df = pd.DataFrame(newtop10).set_index('Label')
print(newtop10_df)


# In[86]:


#Merging Dataframes for Top10 and NewTop10

inner_merge = pd.merge(left = newtop10_df, right = top10_df, left_on = ['Country', 'Label'], right_on = ['Country', 'Label'])
print(inner_merge)


# In[116]:


plt.subplots(figsize=(15,4))

plt.plot(inner_merge['Country'],inner_merge['2016 Health spending % gdp'])
plt.plot(inner_merge['Country'],inner_merge['2019 Health spending % gdp'])

plt.xlabel('Country')
plt.ylabel('% GDP Spend')
plt.title('Health Spending as % of GDP (2016 - 2019)')


# In[125]:


#Looking at Ireland specifically

year = [2016, 2019]
GDP_spend = [7.4, 6.7]
sns.barplot (x = year, y = GDP_spend)
plt.title = 'Ireland % GDP Spend on Health (2016-2019)'
plt.show()


# In[97]:


#Final API import for analysis: Covid-19 cases... to be continued!

import requests
covid19 = requests.get('https://api.covid19api.com/summary')
covid19 = covid19.json()
pd.DataFrame(covid19['Countries'])
covid19cases = pd.DataFrame(covid19['Countries'])


# In[108]:


covid19cases.describe()
covid19cases.info()
covid19cases.head(5)

