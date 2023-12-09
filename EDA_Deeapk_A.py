#!/usr/bin/env python
# coding: utf-8

# <h1 style='color:Blue'>EDA OF BANK LOANS</h1>

# <h2 style = "color:Brown"> Importing Required Libraries</h2>

# In[101]:


# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')


# <h2 style = "color:Brown"> Reading and Understanding the Dataset Application_Data</h2>

# In[102]:


appda = pd.read_csv("application_data.csv")
appda.head()


# In[103]:


appda.info(verbose = True, null_counts = True)


# In[104]:


appda.shape


# In[105]:


appda.info('all')


# In[106]:


appda.describe()


# In[107]:


appda.columns


# ## Data checking and Missing values
# 

# In[99]:


# Funcion to get null value

def null_percentage(df):
    output = round(df.isnull().sum()/len(df.index)*100,2)
    return output


# In[100]:


# Missing values of all columns

Na_col = null_percentage(appda)
Na_col


# In[9]:


# Finding out columns with only null values

Na_col = Na_col[Na_col>0]
Na_col


# In[10]:


Na_col.count


# In[11]:


# Visualizing Null values of columns in graph

plt.figure(figsize = (10,4), dpi=100)
Na_col.plot(kind = "bar")
plt.title("Null values in columns")
plt.xlabel('Percentage of Null')
plt.show()


# <h2 style = "color:Brown"> Remove columns with > 50% missing data</h2>

# In[12]:


# Taking out columns with >50%

Na_col50 = Na_col[Na_col>50]
print("Number of columns with null value > 50% :", len(Na_col50.index))
print(Na_col50)


# In[13]:


appda.shape


# In[14]:


appda.head()


# In[15]:


appda.columns


# In[16]:


# Columns with null values <15%

Na_col15 = Na_col[Na_col<15]
print("Number of columns with null value < 15% :", len(Na_col15.index))
print(Na_col15)


# In[17]:


Na_col15.index


# In[18]:


# Identifying unique values with columns <15%

appda[Na_col15.index].nunique().sort_values(ascending=False)


# From the above we can see that first two (EXT_SOURCE_2, AMT_GOODS_PRICE) are continous variables and remaining are catagorical variables

# In[19]:


# Continous varibale

plt.figure(figsize=(10,2))
sns.boxplot(appda['EXT_SOURCE_2'])
plt.show()


# In[20]:


plt.figure(figsize=(10,2))
sns.boxplot(appda['AMT_GOODS_PRICE'])
plt.show()


# Observation from Boxplots:
# 
# For 'EXT_SOURCE_2' no outliers present. So data is rightly present.
# For 'AMT_GOODS_PRICE' outlier present in the data. so need to impute with median value: 4

# In[21]:


for col in appda.columns:
    print(col)


# ## Now removing the columns from the data set which are unused for better analysis
# 

# In[23]:


col_unused = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE','FLAG_PHONE', 'FLAG_EMAIL',
          'REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL','CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
          'REGION_RATING_CLIENT_W_CITY','FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4',
          'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
          'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
          'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
          'FLAG_DOCUMENT_21','EXT_SOURCE_2','EXT_SOURCE_3','YEARS_BEGINEXPLUATATION_AVG','FLOORSMAX_AVG','YEARS_BEGINEXPLUATATION_MODE',
          'FLOORSMAX_MODE','YEARS_BEGINEXPLUATATION_MEDI','FLOORSMAX_MEDI','TOTALAREA_MODE','EMERGENCYSTATE_MODE']


# In[24]:


appda.drop(labels = col_unused, axis=1, inplace = True)


# In[25]:


appda.head()


# In[26]:


appda.shape


# In[27]:


# Imputing the value'XNA' which means not available for the column 'CODE_GENDER'

appda.CODE_GENDER.value_counts()


# <h2 style = "color:Brown"> XNA values are very low and Female is the majority. So lets replace XNA with gender 'F'</h2>

# In[28]:


appda.loc[appda.CODE_GENDER == 'XNA', 'CODE_GENDER'] = 'F'


# In[29]:


appda.CODE_GENDER.value_counts()


# In[31]:


# checking the CODE_GENDER 

appda.CODE_GENDER.head(10)


# In[32]:


appda.info('all')


# In[33]:


# Casting variable into numeric in the dataset

numerical_columns=['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','REGION_POPULATION_RELATIVE',
                 'DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START',
                 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
                'DAYS_LAST_PHONE_CHANGE']

appda[numerical_columns] = appda[numerical_columns].apply(pd.to_numeric)
appda.head(10)


# In[34]:


# 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH'


# In[35]:


# Age/Days columns are in -ve which needs to be converted to +ve value

appda.DAYS_BIRTH = appda.DAYS_BIRTH.abs()
appda.DAYS_EMPLOYED = appda.DAYS_EMPLOYED.abs()
appda.DAYS_REGISTRATION = appda.DAYS_REGISTRATION.abs()
appda.DAYS_ID_PUBLISH = appda.DAYS_ID_PUBLISH.abs()


# In[36]:


appda.head()


# In[37]:


appda.tail()


# In[39]:


# Checking outliers of numerical_column

appda[numerical_columns].describe()


# In[40]:


# Now lets check box plot for 'CNT_CHILDREN', 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','DAYS_EMPLOYED', 'DAYS_REGISTRATION' 

plt.figure(figsize = (12,3))
sns.boxplot(appda['CNT_CHILDREN'])
plt.show()


# 1st quartile is missing for CNT_CHILDREN which means most of the data are present in the 1st quartile.

# In[41]:


plt.figure(figsize = (12,3))
sns.boxplot(appda['AMT_INCOME_TOTAL'])
plt.show()


# In AMT_INCOME_TOTAL only single high value data point is present as outlier
# 

# In[42]:


plt.figure(figsize = (12,3))
sns.boxplot(appda['AMT_CREDIT'])
plt.show()


# AMT_CREDIT has little bit more outliers

# In[43]:


plt.figure(figsize = (12,3))
sns.boxplot(appda['AMT_ANNUITY'])
plt.show()


# 1st quartiles and 3rd quartile for AMT_ANNUITY is moved towards first quartile.
# 

# In[44]:


plt.figure(figsize = (12,3))
sns.boxplot(appda['DAYS_EMPLOYED'])
plt.show()


# Same with this too 1st quartiles and 3rd quartile for DAYS_EMPLOYED is stays first quartile.
# 

# In[45]:


plt.figure(figsize = (12,3))
sns.boxplot(appda['DAYS_REGISTRATION'])
plt.show()


# Same with this too 1st quartiles and 3rd quartile for DAYS_EMPLOYED is stays first quartile.
# From above box plots we found that numeric columns have outliers

# In[46]:


# Creating bins for continous variable categories column 'AMT_INCOME_TOTAL', 'AMT_GOODS_PRICE' and 'AMT_CREDIT'

bins = [0,100000,200000,300000,400000,500000,10000000000]
slots = ['<100000', '100000-200000','200000-300000','300000-400000','400000-500000', '500000 and above']

appda['AMT_INCOME_RANGE'] = pd.cut(appda['AMT_INCOME_TOTAL'], bins = bins, labels=slots)


# In[47]:


bins = [0,100000,200000,300000,400000,500000,10000000000]
slots = ['<100000', '100000-200000','200000-300000','300000-400000','400000-500000', '500000 and above']

appda['AMT_CREDIT'] = pd.cut(appda['AMT_CREDIT'], bins = bins, labels=slots)


# In[48]:


ins = [0,100000,200000,300000,400000,500000,10000000000]
slots = ['<100000', '100000-200000','200000-300000','300000-400000','400000-500000', '500000 and above']

appda['AMT_GOODS_PRICE'] = pd.cut(appda['AMT_GOODS_PRICE'], bins = bins, labels=slots)


# In[49]:


appda.AMT_GOODS_PRICE.head()


# In[50]:


appda.AMT_CREDIT.head()


# In[51]:


appda.AMT_INCOME_RANGE.head()


# In[52]:


appda.head()


# Analysis

# In[53]:


# Dividing the dataset into two dataset of Target=1(client with payment difficulties) and Target=0(all other)


# In[54]:


Target0 = appda.loc[appda.TARGET == 0]
Target1 = appda.loc[appda.TARGET == 1]


# In[55]:


appda.TARGET.head()


# In[56]:


# Calculating Imbalance percentage
    
# Since the majority is target0 and minority is target1

Imb = round(len(Target0)/len(Target1),2)

print('Imbalance Ratio:', Imb)


# In[57]:


appda.columns


# ## Univariate Analysis

# In[59]:


sns.set_style('whitegrid')
sns.set_context('notebook')
plt.rcParams["axes.labelsize"] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.titlepad'] = 12


# In[60]:


flow = ['AMT_INCOME_RANGE','CODE_GENDER', 'NAME_INCOME_TYPE','NAME_CONTRACT_TYPE']
plt.figure(figsize = (20, 15))

for i in enumerate(flow):
    plt.subplot(2, 2, i[0]+1)
    plt.subplots_adjust(hspace=0.5)
    sns.countplot(x = i[1], hue = 'TARGET', data = appda)
    
    plt.rcParams['axes.titlesize'] = 16
    
    plt.xticks(rotation = 90)
    plt.yscale('log')


# <h2 style = "color:Brown"> Reading Previous_application Dataset </h2> 

# In[69]:


# Reading the previous_application csv file

preapp = pd.read_csv('previous_application.csv')
preapp.head()


# In[70]:


# Number of rows and columns in previous application data frame

preapp.shape


# In[72]:


# Knowing the previous application data frame info

preapp.info()


# In[73]:


# describing the previous application data frame

preapp.describe()


# In[74]:


# Finding out null values

Nu_col = null_percentage(preapp)
Nu_col.head()


# In[75]:


# Removing null values >0

Nu_col0 = Nu_col[Nu_col>0]
Nu_col0


# In[76]:


# Now removing null values <50

Nu_col50 = Nu_col[Nu_col<50]
Nu_col50


# In[77]:


# Merging both the dataframes (application_data, previous_application)

da_combi = pd.merge(left = appda, right = preapp, how='inner', on ='SK_ID_CURR', suffixes ='_x')
da_combi.head()


# In[78]:


da_combi.shape


# In[79]:


da_combi.columns


# In[80]:


da_combi.info('all')


# In[81]:


# Performing univarite analysis


# In[82]:


# Purpose of loan


plt.figure(figsize=(10,10),dpi = 300)
plt.xticks(rotation=90)
plt.title('Purpose of loan')
sns.set_style('darkgrid')
ax = sns.countplot(data = da_combi, y= 'NAME_CASH_LOAN_PURPOSE', order=da_combi['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS') 
plt.show()


# Observation
# - Most of loan rejection was from 'repairs'

# In[83]:


# Purpose of loan with TARGET column

plt.figure(figsize=(10,10),dpi = 300)
plt.xticks(rotation=90)
plt.title('Purpose of loan with TARGET column')
sns.set_style('darkgrid')
ax = sns.countplot(data = da_combi, y= 'NAME_CASH_LOAN_PURPOSE', order=da_combi['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET') 
plt.show()


# <h1 style = "color:Blue"> Data Cleaning Completed </h1>

# <h2 style = "color:Green"> Data Analysis </h2>
# 

# # Observation
# - Most of loan rejection was from 'repairs'

# In[108]:


#Some columns have negative values which do not make sense as per their context, 
#therfore converting their values to absolute values:

appda['DAYS_BIRTH'] = abs(appda['DAYS_BIRTH'])
appda['DAYS_ID_PUBLISH'] = abs(appda['DAYS_ID_PUBLISH'])
appda['DAYS_ID_PUBLISH'] = abs(appda['DAYS_ID_PUBLISH'])


# In[109]:


appda.head()


# In[112]:


# Creating bins for income amount (AMT_INCOME_TOTAL)

bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']
appda ['AMT_INCOME_RANGE'] = pd.cut(appda['AMT_INCOME_TOTAL'], bins=bins, labels=slot)


# In[113]:


appda["AMT_INCOME_RANGE"].head()


# In[114]:


#Creating bins for Credit amount (AMT_CREDIT)

bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

appda['AMT_CREDIT_RANGE'] = pd.cut(appda['AMT_CREDIT'], bins=bins, labels=slots)


# In[115]:


appda["AMT_CREDIT_RANGE"].head()


# In[116]:


#Dividing the dataset into two datasets of  target=1 (client with payment difficulties) and target=0 (all other cases)
#We will be using these two datasets for few comparisons
target0 = appda[appda["TARGET"]==0]
target1 = appda[appda["TARGET"]==1]


# In[117]:


print(target0.shape)
print(target1.shape)


# In[118]:


#Calculating Imbalance percentage for target 0 and target 1. 
#Visualizing the above result in a pie plot

total = len(appda["TARGET"])
explode = [0, 0.05]

def my_fmt(x):
    return '{:.2f}%\n({:.0f})'.format(x, total*x/100) #to print both the percentage and value together

plt.figure(figsize = [6, 6])
plt.title("Imbalance between target0 and target1")
appda["TARGET"].value_counts().plot.pie(autopct = my_fmt, colors = ["teal", "gold"], explode = explode)

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# 8.78% of clients are clients with payment difficulties. 91.21% of clients fall under the 'all other cases' category.

# ## Plotting graphs for Target0 (Customers with no payment difficulties)

# In[119]:


# Count plotting in logarithmic scale

#creating a function for plotting (to reduce repetitive code)
def functionPlot(df, col, title, xtitle, ytitle, hue = None):
    
    sns.set_style("white")
    sns.set_context("notebook")    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation = 45)
    plt.yscale('log')
    plt.title(title)

    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue, palette='deep') 
    ax.set(xlabel = xtitle, ylabel = ytitle)    
    plt.show()


# In[121]:


# PLotting for income range (AMT_INCOME_RANGE) (Segregated based on gender)

functionPlot(target0, col='AMT_INCOME_RANGE', title='Distribution of Income Range (Repayers)', hue='CODE_GENDER', xtitle = "Income Range", ytitle = "Count")


# <h1 style = "color:Pink"> Inferences </h1>
# 
# 1. In majority of the cases, female counts are higher than male.
# 2. Income range from 100000 to 200000 is having more number of credits.
# 3. In the slots 250000-275000 and 375000-400000, the count for both males and females are almost the same

# In[122]:


# Plotting for Income type (NAME_INCOME_TYPE) (Segregated based on gender)

functionPlot(target0, col='NAME_INCOME_TYPE', title='Distribution of Income type (Repayers)', hue='CODE_GENDER', xtitle = "Income type of the client", ytitle = "Count")


# <h1 style = "color:Pink"> Inferences </h1>
# 1. Working people make up most of the clients.
# 2. Gender distribution amongst students is almost the same.
# 3. There is a stark decerease in the amount of clients if they are not working, commerical associate or state servants.
# 4. There are more female clients in the top 3 categories of income type.

# In[124]:


# Plotting for Contract type (NAME_CONTRACT_TYPE) (Segregated based on gender)

functionPlot(target0, col='NAME_CONTRACT_TYPE', title='Distribution of contract type (Segregated based on gender)', hue='CODE_GENDER', xtitle = "Type of Contract", ytitle = "Count")


# <h1 style = "color:Pink"> Inferences </h1>
# 
# Cash loans clearly has more clients per credit compared to revolving loans. In both cases, there are female clients than male clients.

# In[125]:


# Plotting for Organization type in logarithmic scale 

sns.set_style('darkgrid')
sns.set_context('notebook')
plt.figure(figsize = [15,30])

plt.title("Distribution of Organization type for target0 (All other cases)")
plt.xscale('log')

sns.countplot(data=target0,y='ORGANIZATION_TYPE',order=target0['ORGANIZATION_TYPE'].value_counts().index, palette='deep')

plt.show()


# <h1 style = "color:Pink"> Inferences </h1>
# 
# 1. Clients which have applied for credits are from most of the organization type ‘Business entity Type 3’ , ‘Self employed’, ‘Other’ , ‘Medicine’, ‘Government’ and 'Business entity type2'.
# 2. Fewer clients are from Industries type 8, type 5, Industry: type 13, Trade: type4, Religion, Industry type 6 and type 10.

# ## Plotting graphs for Target1 (Customers with payment difficulties)

# In[127]:


# PLotting for income range (Segregated based on gender)

functionPlot(target1, col='AMT_INCOME_RANGE', title='Distribution of Income Range (Customers with payment difficulties)', hue='CODE_GENDER', xtitle = "Income Range", ytitle = "Count")


# <h1 style = "color:Pink"> Inferences </h1>
# 
# 1. Income range from 100000 to 200000 is having the highest number of credits.
# 2. Very less count for income range 400000 and above.
# 3. On average, there are more number of male clients where the number of credits are less.

# In[129]:


# Plotting for Income type (Segregated based on house ownership)

functionPlot(target1, col='NAME_INCOME_TYPE', title='Distribution of Income type (Customers with payment difficulties)', hue='FLAG_OWN_REALTY', xtitle = "Income type of the client", ytitle = "Count")


# <h1 style = "color:Pink"> Inferences </h1>
# 
# 1. Working customers, obviously, have a higher count.
# 2. As we can see, most customers do have their own property (house or a flat) but a large number of customers can be stated as otherwise.

# In[130]:


# Plotting for Contract type (Segregated based on education level)

functionPlot(target1, col='NAME_CONTRACT_TYPE', title='Distribution of Contract Type (Customers with payment difficulties)', hue='NAME_EDUCATION_TYPE', xtitle = "Type of contract", ytitle = "Count")


# <h1 style = "color:Pink"> Inferences </h1>
# 
# 1. Cash loans, as we can see, are preferred by clients of all education backgrounds with an overwhelming majority.
# 2. People with only an academic degree do not prefer revolving loans at all.

# In[131]:


# Plotting for Organization type

sns.set_style('darkgrid')
sns.set_context('notebook')
plt.figure(figsize=(15,30))

plt.title("Distribution of Organization type of Clients with payment difficulties (Target1)")

plt.xticks(rotation=90)
plt.xscale('log')

sns.countplot(data = target1, y = 'ORGANIZATION_TYPE', order = target1['ORGANIZATION_TYPE'].value_counts().index, palette='deep')

plt.show()


# <h1 style = "color:Pink"> Inferences </h1>
# 
# 1. As compared to the clients with NO payment difficulties, clients WITH payment difficulties have the 'construction' business type in the top 5 count replacing the 'medicine' business type.
# 
# 2. Most of the business types are the same as clients with NO payment difficulties, except we have the business type 'Transport: type1' in the case of clients WITH payment difficulties which wasn't present before.

# # More Analysis to find patterns:

# In[132]:


#Separting the categorical columns from the numerical ones
obj_dtypes = [i for i in appda.select_dtypes(include=np.object).columns if i not in ["type"] ]
num_dtypes = [i for i in appda.select_dtypes(include = np.number).columns if i not in ['SK_ID_CURR'] + [ 'TARGET']]


# In[133]:


#Print categorical columns
for x in range(len(obj_dtypes)): 
    print(obj_dtypes[x])


# In[134]:


#Print numerical columns
for x in range(len(num_dtypes)): 
    print(num_dtypes[x])


# ### Distribution of Target variable¶
# Target variable:
# 
# 1 - client with payment difficulties
# 
# 0 - all other cases

# In[135]:


plt.figure(figsize = [20, 6])

plt.suptitle("Distribution of clients with difficulties and all other cases")

plt.subplot(1,2,1)
ax = appda["TARGET"].value_counts().plot(kind = "barh", colormap = "summer")

plt.subplot(1,2,2)
appda["TARGET"].value_counts().plot.pie(autopct = "%.2f%%", startangle = 60, colors = ["teal", "gold"])


for i,j in enumerate(appda["TARGET"].value_counts().values):
    ax.text(.7, i, j, weight = "bold")

plt.show()


# <h1 style = "color:Pink"> Inferences </h1>
# 
# 8.79% (18547) out of total client population (192573) have difficulties in repaying loans.

# ## Concatenating applicationData and previousApplication

# In[136]:


## Concatenating applicationData and previousApplication
#Joining the two data sets by a common column (SK_ID_CURR) for further comparison and analysis

data = pd.merge(appda, preapp, on = 'SK_ID_CURR', how = 'inner')
data.sort_values(by = ['SK_ID_CURR','SK_ID_PREV'], ascending = [True, True], inplace = True)


# In[137]:


data.head(10) #THE COMBINED DATASET


# ## Distribution in Contract types in data (Combined dataset)

# In[138]:


#Distribution of Contract type
plt.figure(figsize = [14, 6])
plt.subplot(1,2,1)
plt.title("distribution of contract types in data (combined dataset)")
data["NAME_CONTRACT_TYPE_x"].value_counts().plot.pie(autopct = "%1.0f%%", startangle = 60, colors = ["teal", "gold"])

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# The percentage of revolving loans and cash loans are 8% & 92%.

# ### Point to infer from the graph
# In the applicationData file, we saw females had 61% and males had 39% but now in the combined dataset we see:-
# Females: 62%
# Males: 38%

# ### Distribution of Contract type by target (repayment status)

# In[141]:


fig = plt.figure(figsize = [20, 6])

plt.subplot(1,2,1)
plt.title("Distribution of Contract type by target variable", weight = "bold")

sns.countplot("NAME_CONTRACT_TYPE_x", hue = "TARGET", data = data, palette="deep")
plt.xlabel("Contract type", weight = "bold")
plt.ylabel("Count", weight = "bold")
plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# Both set of clients (Target 0 and target 1) prefer cash loans over revolving loans with overwhelming numbers

# ## Distribution of Gender by target (repayment status)

# In[143]:


fig = plt.figure(figsize = [20, 6])

plt.subplot(1,2,1)
plt.title("Distribution of Gender by target variable", weight = "bold")

sns.countplot("CODE_GENDER", hue = "TARGET", data = data, palette="deep")
plt.xlabel("Contract type", weight = "bold")
plt.ylabel("Count", weight = "bold")
plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# 1. Clearly, female clients are the best repayers of their loan (almost double the amount of males).
# 2. Amount of defaulters in both genders are almost equally distributed.

# ### Distribution of client owning a car and by target.

# In[144]:


fig = plt.figure(figsize = [15,6])
explode = (0, 0.1)

plt.subplot(1,2,1)
plt.title("Distribution of Client by car ownership", weight = "bold")
data["FLAG_OWN_CAR"].value_counts().plot.pie(autopct = "%1.0f%%", colors=["teal", "gold"], explode = explode, startangle = 60)

plt.subplot(1,2,2)
plt.title("Distribution of Client by car ownership based on repayment status", weight = "bold")
data[data["FLAG_OWN_CAR"] == "Y"]["TARGET"].value_counts().plot.pie(autopct = "%1.0f%%", colors=["brown","grey"], explode = explode)

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 1st pie plot : Only 38% of clients own a car .
# 
# 2nd pie plot : Only 8% of clients who own a car have difficulty in payments 

# ### Distribution of client owning a house or flat and by target¶

# In[146]:


#FLAG_OWN_REALTY - Flag if client owns a house or flat
fig = plt.figure(figsize = [15,6])
explode = [0, 0.05]

plt.subplot(1,2,1)
plt.title("Distribution of Client by house ownership", weight = "bold")
data["FLAG_OWN_REALTY"].value_counts().plot.pie(autopct = "%1.0f%%", colors = ["teal","gold"], startangle = 60, explode = explode)


plt.subplot(1,2,2)
plt.title("Distribution of client by house ownership based on repayment status", weight = "bold")
data[data["FLAG_OWN_REALTY"] == "Y"]["TARGET"].value_counts().plot.pie(autopct = "%1.0f%%", colors = ["brown","grey"], explode = explode)

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# SUBPLOT 1 : 71% of clients own a house or a flat.
# 
# SUBPLOT 2 : Out of all the clients who own a house, 9% of clients have difficulty in making payments.

# ### Distribution of Number of children and family members of client by repayment status (Based on target).

# In[147]:


# CNT_CHILDREN - Number of children the client has.
# CNT_FAM_MEMBERS - How many family members does client have.

fig = plt.figure(figsize= [12, 15])

plt.subplot(2,1,1)
plt.title(" Distribution of Number of children client has  by repayment status", weight = "bold")
ax = sns.countplot(data["CNT_CHILDREN"], hue = data["TARGET"])
ax.set(xlabel = "Number of Children", ylabel = "Count of Clients")
plt.legend(loc = "right")

plt.subplot(2,1,2)
plt.title(" Distribution of Number of family members client has  by repayment status", weight = "bold")
ax = sns.countplot(data["CNT_FAM_MEMBERS"], hue = data["TARGET"])
ax.set(xlabel = "Number of Family Members", ylabel = "Count of Clients")
plt.legend(loc = "right")

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# Subplot1:
# 1. The majority as per both cases of repayment status, have zero children.
# 2. Clients with more than 2 children do not have difficulty in making payments.
# 3. Clients with 0 children have the majority in terms of having difficulty in making payments.
# 
# Subplot2:
# 1. Clients with 2 family members living together are in high numbers as per both cases of repayment status
# 2. Also, from point 1, the majority of clients having difficulty in payments have 2 family members

# ### Distribution of Suite type

# In[148]:


#NAME_TYPE_SUITE - Who was accompanying client when he was applying for the loan.

plt.figure(figsize = [18, 12])

plt.subplot(1,2,1)
plt.title("Distribution of Suite type", weight = "bold")
ax = sns.countplot(y = data["NAME_TYPE_SUITE_y"], palette = "deep", order = data["NAME_TYPE_SUITE_y"].value_counts().index[:5])
ax.set(xlabel = "Count", ylabel = "Who accompanied Client")

plt.subplot(1,2,2)
plt.title("Distribution of Suite type by target (repayment status)", weight = "bold")
ax = sns.countplot(y = data["NAME_TYPE_SUITE_y"], palette="deep", hue=data["TARGET"], order = data["NAME_TYPE_SUITE_y"].value_counts().index[:5])
ax.set(xlabel = "Count", ylabel = "")
plt.legend(loc = "right")

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# Note: Missing data was labelled as 'missing' during the data cleaning process so we can ignore it.
# 
# 1. Majority of the clients are (in both cases of repayment status) unaccompanied (without anyone to help/guide them)
# 2. Least amount of clients are in the company of their children.
# 

# ### Distribution of client income type

# In[149]:


#NAME_INCOME_TYPE: Client's income type
plt.figure(figsize = [22, 8])

plt.subplot(1,2,1)
plt.title("Distribution of client income type",  weight = "bold")
ax = sns.countplot(y = data["NAME_INCOME_TYPE"], palette = "deep", order = data["NAME_INCOME_TYPE"].value_counts().index[:4])
ax.set(xlabel = "Count", ylabel = "Income Type")

plt.subplot(1,2,2)
plt.title("Distribution of client income  type by target (repayment status)",  weight = "bold")
ax = sns.countplot(y = data["NAME_INCOME_TYPE"],  hue = data["TARGET"], palette="deep", order = data["NAME_INCOME_TYPE"].value_counts().index[:4])
ax.set(xlabel = "Count", ylabel = "")

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# 1. Most clients as per both cases of repayment status, are working.
# 2. Conversely, the least amount of clients are pensioners (retired clients)

# ### Distribution of Education type by repayment status
# 

# In[150]:


#NAME_EDUCATION_TYPE: Level of education the client

explode = [0, 0.05, 0.08, 0.08, 0.08]

plt.figure(figsize = [20, 8])
plt.subplot(1,2,1)
plt.title("Distribution of Education for Repayers",  weight = "bold")
data[data["TARGET"] == 0]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=12, autopct = "%1.0f%%", explode = explode)

plt.subplot(1,2,2)
plt.title("Distribution of Education for Defaulters",  weight = "bold")
data[data["TARGET"] == 1]["NAME_EDUCATION_TYPE"].value_counts().plot.pie(fontsize=12, autopct = "%1.0f%%", explode = explode)

plt.show()


# ### Point to infer from the graph
# 
# 1. Clients who default are proportionally 9% higher compared to clients who do not default (for clients with education as secondary).
# 2. In the higher education category, clients who default are 8% fewer.
# 3. In both cases of repayment status, lower secondary and academic degree categories are the minority.

# ### Average Earnings by different professions based on target (repayment status)

# In[151]:


#creating a dataframe grouped by the desired columns
data2 = data.groupby(['TARGET','NAME_INCOME_TYPE'])['AMT_INCOME_TOTAL'].mean().reset_index().sort_values(by='AMT_INCOME_TOTAL', ascending = False)
fig = plt.figure(figsize = [16, 8])

ax = sns.barplot('NAME_INCOME_TYPE','AMT_INCOME_TOTAL', data = data2, hue='TARGET', palette = "deep")
plt.title("Average Earnings by Profession (Grouped by repayment status)", weight = "bold")

plt.xlabel("Profession of the client", weight = "bold")
plt.ylabel("Income", rotation = 0, weight = "bold")
plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 1. In both cases of repayment status, commerical associate clients are the highest earners.
# 2. Clients who are on maternity leave (therefore, female clients) have difficulty in making payments
# 3. Pensioners and students do not have any difficulties in repayments.
# 4. There are almost an equal number of clients under the working category who repay and default.

# ### Distribution of Education type by loan repayment status
# 

# In[152]:


#NAME_FAMILY_STATUS - Family status of the client

plt.figure(figsize = [16, 8])
plt.subplot(1,2,1)
plt.title("Distribution of Family status for Repayers (Target0)",  weight = "bold")
data[data["TARGET"]==0]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%")

plt.subplot(1,2,2)
plt.title("Distribution of Family status for Defaulters (Target1)", weight = "bold")
data[data["TARGET"]==1]["NAME_FAMILY_STATUS"].value_counts().plot.pie(autopct = "%1.0f%%")

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# 1. There's a difference of -4% in married clients who have difficulty in making payments.
# 2. Family status for both cases of repayment status have an almost evenly distributed family status (family members living with the client)

# ### Distribution of credit amount and housing type

# In[154]:


# Box plotting for Credit amount and Housing type (Segregated by repyament status (Target))

plt.figure(figsize = [16,12])
plt.title('Credit amount vs Housing type',  weight = "bold")

sns.barplot(data = data, x = 'NAME_HOUSING_TYPE', y = 'AMT_CREDIT_x', hue = 'TARGET', palette = "deep")
plt.xticks(rotation = 0)

plt.xlabel("Housing type", weight = "bold")
plt.ylabel("Credit Amount", weight = "bold", rotation = 0)

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# 1. Clients with office, co-op, municipal aparments have the highest repayers.
# 2. Clients living with parents or in a parents' aparment have the least amount of repayers and defaulters.

# ### Distribution of Loan purpose (Segregated by repyament status (Target))

# In[155]:


#Using log scale for distribution
sns.set_style('dark')
sns.set_context('notebook')

plt.figure(figsize = [15, 30])
plt.xscale('log')
plt.title('Distribution of purposes with target (Repayment status)',  weight = "bold")

ax = sns.countplot(data = data, y = 'NAME_CASH_LOAN_PURPOSE', order = data['NAME_CASH_LOAN_PURPOSE'].value_counts().index, hue = 'TARGET', palette = 'deep') 
plt.ylabel("Purpose of Loan", weight = "bold")
plt.xlabel("Count", weight = "bold")

plt.legend(loc = "right")
plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 
# 1. Repair purposes are on top with most defaulters and repayers.
# 2. Proportion wise, there are high amount of repayers when the client refuses to name the purpose of the loan. Although such clients are rare.

# ### Distribution of contract status

# In[156]:


#Using log scale for distribution
sns.set_style('dark')
sns.set_context('notebook')

plt.figure(figsize = [15, 30])
plt.xscale('log')
plt.title('Distribution of purposes with target (Repayment status)',  weight = "bold")

ax = sns.countplot(data = data, y = 'NAME_CASH_LOAN_PURPOSE', order = data['NAME_CASH_LOAN_PURPOSE'].value_counts().index, hue = 'NAME_CONTRACT_STATUS', palette = 'deep') 
plt.ylabel("Purpose of Loan", weight = "bold")
plt.xlabel("Count", weight = "bold")

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 1. Most rejection of loans is when the purpose of the client is based on Repairs.
# 2. For education purposes we have equal number of approvals and refusals.

# # TOP10 Correlation variables

# In[157]:


repayerData = data[data['TARGET'] == 0]
defaulterData = data[data['TARGET'] == 1]


# In[158]:


#to find the most correlated columns (positive and negative)
repayerData.corr().unstack().sort_values(ascending = False).drop_duplicates()


# ## From the above output, the top10 correlated columns are: (double click to view in proper format)
# 
# 1. OBS_30_CNT_SOCIAL_CIRCLE    OBS_60_CNT_SOCIAL_CIRCLE        1.00
# 2. AMT_CREDIT_y                AMT_APPLICATION                 0.97
# 3. DAYS_TERMINATION            DAYS_LAST_DUE                   0.93
# 4. CNT_FAM_MEMBERS             CNT_CHILDREN                    0.90
# 5. REG_REGION_NOT_WORK_REGION  LIVE_REGION_NOT_WORK_REGION     0.88
# 6. DEF_30_CNT_SOCIAL_CIRCLE    DEF_60_CNT_SOCIAL_CIRCLE        0.87
# 7. AMT_GOODS_PRICE_y           AMT_CREDIT_y                    0.86
# 8. AMT_APPLICATION             AMT_GOODS_PRICE_y               0.85
# 9. REG_CITY_NOT_WORK_CITY      LIVE_CITY_NOT_WORK_CITY         0.83
# 10. AMT_CREDIT_y                AMT_ANNUITY_y                  0.81

# In[160]:


#making a dataframe with only the columns with high correlation
top10_CorrTarget0 = repayerData[["OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "AMT_APPLICATION", "DAYS_TERMINATION", "DAYS_LAST_DUE", "CNT_FAM_MEMBERS", "CNT_CHILDREN", "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "AMT_GOODS_PRICE_y", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY", "AMT_CREDIT_y", "AMT_ANNUITY_y"]].copy()


# In[161]:


top10_CorrTarget0.shape


# # Visually showcasing the top10 correlated columns through a heatmap

# In[162]:


#Visually showcasing the top10 correlated columns through a heatmap

corr_target0 = top10_CorrTarget0.corr()

plt.figure(figsize = [15, 15])
sns.heatmap(data = corr_target0, cmap="Blues", annot=True)

plt.show()


# <h1 style = "color:Green"> Inferences </h1>
# 1. AMT_GOODS_PRICE and AMT_APPLICATION have a high correlation, which means the more credit the client asked for previously is proportional to the goods price that the client asked for previously.
# 2. AMT_ANNUITY and AMT_APPLICATION also have a high correlation, which means the higher the loan annuity issued, the higher the goods price that the client asked for previously.
# 3. If the client's contact address does not match the work address, then there's a high chance that the client's permanent address also does not match the work address.
# 4. First due of the previous application is highly correlated with Relative to the expected termination of the previous application
# 5. CNT_CHILDREN and CNT_FAM_MEMBERS are highly correlated which means a client with children is higly likely to have family members as well.

# In[163]:


defaulterData.corr().unstack().sort_values(ascending = False).drop_duplicates()


# # From the above output, the top10 correlated columns are: (double click to view in proper format)
# 
# 1. OBS_60_CNT_SOCIAL_CIRCLE     OBS_30_CNT_SOCIAL_CIRCLE       1.00
# 2. AMT_APPLICATION              AMT_CREDIT_y                   0.97
# 3. DAYS_TERMINATION             DAYS_LAST_DUE                  0.95
# 4. CNT_FAM_MEMBERS              CNT_CHILDREN                   0.90
# 5. LIVE_REGION_NOT_WORK_REGION  REG_REGION_NOT_WORK_REGION     0.87
# 6. DEF_30_CNT_SOCIAL_CIRCLE     DEF_60_CNT_SOCIAL_CIRCLE       0.86
# 7. AMT_CREDIT_y                 AMT_ANNUITY_y                  0.83
# 8. LIVE_CITY_NOT_WORK_CITY      REG_CITY_NOT_WORK_CITY         0.78
# 9. AMT_ANNUITY_y                AMT_GOODS_PRICE_y              0.76
# 10. AMT_ANNUITY_x                AMT_CREDIT_x                  0.74

# In[164]:


#Adding the top10 correlated columns into a new dataframe:
top10_CorrTarget1 = data[["OBS_60_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE", "AMT_APPLICATION", "AMT_CREDIT_y", "DAYS_TERMINATION", "DAYS_LAST_DUE", "CNT_FAM_MEMBERS", "CNT_CHILDREN", "LIVE_REGION_NOT_WORK_REGION", "REG_REGION_NOT_WORK_REGION", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "AMT_ANNUITY_y", "LIVE_CITY_NOT_WORK_CITY", "REG_CITY_NOT_WORK_CITY", "AMT_GOODS_PRICE_y", "AMT_ANNUITY_x", "AMT_CREDIT_x"]].copy()


# In[165]:


corr_target1 = top10_CorrTarget1.corr()

plt.figure(figsize = [15, 15])
sns.heatmap(data = corr_target1, cmap="Blues", annot=True)
plt.show()


# <h2 style = "color:Green"> Inferences</h2>
# 1. In comparison to the repayer heatmap, AMT_GOODS_PRICE and AMT_APPLICATION have a high correlation here as well, which means the more credit the client asked for previously is proportional to the goods price that the client asked for previously.
# 2. In comparison to the repayer heatmap, AMT_ANNUITY and AMT_APPLICATION also have a high correlation, which means the higher the loan annuity issued, the higher the goods price that the client asked for previously.
# 3. In comparison to the repayer heatmap, If the client's contact address does not match the work address, then there's a high chance that the client's permanent address also does not match the work address.
# 4. Higher the goods price, higher the credit by the client
# 5. First due of the previous application is highly correlated with Relative to the expected termination of the previous application (same as with the repayer heatmap)
# 6. CNT_CHILDREN and CNT_FAM_MEMBERS are highly correlated which means a client with children is higly likely to have family members as well (same as with the repayer heatmap)

# <h2 style = "color:green"> Conclusion from the Analysis </h2>
#  
# Banks must target more on contract type ‘Student’ ,’Pensioner’ and ‘Businessman’ for profitable business
# 
# Banks must focus less on income type ‘Working’ as it is has most number of unsuccessful payments in oreder to get rid of financial loss for the organization
# #### 1. Clients who are Students, Pensioners and Commercial Associates with a housing type such as office/co-op/municipal apartments NEED TO BE TARGETED by the bank for successful repayments. These clients have the highest amount of repayment history.
# 
# #### 2. Female clients on maternity leave should NOT be targeted as they have no record of repayments (therefore they are highly likely to default and targeting them would lead to a loss)
# 
# #### 3. While clients living with parents have the least amount of repayers, they also have the least amount of defaulters. So, in cases where the risk is less, such clients can be TARGETED.
# 
# #### 4. Clients who are working need to be targeted LESS by the bank as they have the highest amount of defaulters.
# 
# #### 5. Clients should NOT be targeted based on their education type alone as the data is very inconclusive.
# 
# #### 6. Banks SHOULD target clients who own a car. 
# 
# #### 7. There are NO repayers/negligible repayers when the contract type is of revolving loan.
# 
# #### 8. Banks SHOULD target more people with no children. 
# 
# #### 9. 'Repairs' purpose of loan is the one with the most defaulters and repayers.  Therefore, clients with very low risk SHOULD be given loans for such purpose to yield high profits.
# 
# #### 10. Banks SHOULD also target female clients as they are the highest repayers (almost as double as males) amongst both the genders.

# In[ ]:




