#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"C:\Users\ravin\OneDrive\Desktop\Portfolio projects\ML\Machine Learning R_27.07.21\Machine Learning Project 63 - Data Scientist's Salary Prediction\glassdoor_jobs.csv")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df.head()


# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.dtypes


# In[8]:


print("shape before removing: {}".format(df.shape))
df.drop(labels= ['Unnamed: 0'],axis=1,inplace=True)
print("shape after removing: {}".format(df.shape))


# Visualizing rating column

# In[9]:


plt.figure(figsize=(6,6))
sns.distplot(df['Rating'])
plt.title("Before Handling -1 value")


# In[10]:


df["Rating"] = df["Rating"].apply(lambda x: np.NaN if x==-1 else x)


# In[11]:


df["Rating"]=df["Rating"].fillna(df["Rating"].mean())


# In[12]:


plt.figure(figsize=(6,6))
sns.distplot(df['Rating'])
plt.title("After Handling -1 value")


# In[13]:


plt.figure(figsize=(5,5))
sns.boxplot(y="Rating",data=df,palette="Set3_r")
plt.title('Boxplot for rating')
plt.ylabel('Rating')


# Visualizing Founded Column

# In[14]:


plt.figure(figsize=(8,8))
sns.distplot(df['Founded'])
plt.title("Before Handling -1 value")


# In[15]:


df["Founded"] = df["Founded"].apply(lambda x: np.NaN if x==-1 else x)


# In[16]:


df["Founded"]=df["Founded"].fillna(df["Founded"].median())


# In[17]:


df['Founded']=df['Founded'].astype('int')


# In[18]:


plt.figure(figsize=(8,8))
sns.distplot(df['Founded'])
plt.title("After Handling -1 value")


# In[19]:


#outliers in Founded column
plt.figure(figsize=(5,5))
sns.boxplot(y="Founded",data=df,palette="Set2_r")
plt.title('Boxplot for Founded')
plt.ylabel('Founded year')


# In[20]:


df['Job Title'].value_counts().nlargest(25)


# In[21]:


def title_cleaner(title):
    if 'data scientist' in title.lower() or 'scientist' in title.lower():
        return 'Data scientist'
    elif 'data engineer' in title.lower():
        return "Data Engineer"
    elif "data analyst" in title.lower():
        return "Data analyst"
    elif 'machine learning' in title.lower():
        return 'MLE'
    elif 'manager' in title.lower():
        return 'Manager'
    elif 'director' in title.lower():
        return 'Director'
    else:
        return 'other'

        
df['job_title']=df['Job Title'].apply(title_cleaner)
df['job_title'].value_counts()


# In[22]:


plt.figure(figsize=(8,8))
p=sns.countplot(x='job_title',data=df)
p.set_xticklabels(p.get_xticklabels(),rotation=45,horizontalalignment='right')


# In[23]:


def senior(title):
    if 'sr.' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
        return 'sr'
    elif 'junior' in title.lower() or 'jr.' in  title.lower():
        return 'jr'
    else:
        return 'other'
df['job_seniority'] = df['Job Title'].apply(senior)
df['job_seniority'].value_counts()


# In[24]:


plt.figure(figsize=(8,8))
g=sns.countplot(x='job_seniority',data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=45,horizontalalignment='right')


# In[25]:


df.drop(labels=['Job Title'],axis=1,inplace=True)


# In[26]:


salary=df['Salary Estimate'].apply(lambda x: x.split("(")[0])
salary


# In[27]:


salary.value_counts()


# In[28]:


salary=salary.apply(lambda x: np.NaN if x=='-1' else x)
salary


# In[29]:


salary=salary.apply(lambda x: x if type(x) == type(5.5) else x.replace('$','').replace('K',''))


# In[30]:


salary


# In[31]:


print('Length of salary:', len(salary.unique()))
salary.unique()


# In[32]:


salary=salary.apply(lambda x: x if type(x) == type(5.5) else x.lower().replace('employer provided salary:',""))
salary.unique()[380:]


# In[33]:


def hourly_to_yearly(lowlimit,upplimit):
    x=lowlimit.strip()
    y=upplimit.strip()
    x=int(int(lowlimit)*45*52/1000)
    y=int(int(upplimit)*45*52/1000)
    return '{}-{}'.format(x,y)
salary=salary.apply(lambda x: x if type(x)==type(5.5) else (hourly_to_yearly(x.lower().replace('per hour','').split('-')[0],x.lower().replace('per hour','').split('-')[1]) if 'per hour' in x.lower() else x))


# In[34]:


salary.unique()[380:]


# Average Salary

# In[35]:


df['Salary']=salary.apply(lambda x: x if type(x)==type(5.5) else (int(x.split('-')[0])+int(x.split('-')[1].strip()))/2)
df['Salary']=df['Salary'].fillna(df['Salary'].median())


# In[36]:


plt.figure(figsize=(6,6))
sns.distplot(df['Salary'])
plt.title("Without NaN values")


# In[37]:


#outliers in salary
plt.figure(figsize=(6,6))
sns.boxplot(y=df['Salary'], data=df,palette='Set1_r')
plt.title("Boxplot for salary")


# In[38]:


df['Company Name'].head()


# In[39]:


df['Company Name']=df['Company Name'].apply(lambda x: x.split("\n")[0])


# In[40]:


plt.figure(figsize=(8,8))
df['Company Name'].value_counts().nlargest(20).plot(kind='barh')
plt.xlabel('Count')


# In[41]:


df['Location'].head()


# In[42]:


df['job_location']=df['Location'].apply(lambda x: x if ',' not in x else x.split(',')[1].strip())


# In[43]:


df['job_location'].head()


# In[44]:


print('Total number of unique locations: {}'.format(len(df['job_location'].unique())))


# In[45]:


plt.figure(figsize=(8,8))
df['job_location'].value_counts().nlargest(20).plot(kind='barh')
plt.xlabel('Count')


# In[46]:


df['Size'].value_counts()


# In[47]:


def size_cleaner(text):
    if '-1' in text.lower():
        return 'Unknown'
    else:
        return text
df['Size']=df['Size'].apply(size_cleaner)


# In[48]:


df['Size'].value_counts()


# In[49]:


plt.figure(figsize=(6,6))
g=sns.countplot(x='Size',data=df)
p=g.set_xticklabels(g.get_xticklabels(),rotation=45,horizontalalignment='right')


# In[50]:


df['Type of ownership'].head()


# In[51]:


df['Type of ownership'].value_counts()


# In[52]:


def company_cleaner(text):
    if 'private' in text.lower():
        return 'Private'
    elif 'public' in text.lower():
        return 'Public'
    elif ('-1' in text.lower()) or ('school / school District' in text.lower()) or ('private practice /firm' in text.lower()) or ('unknown' in text.lower()) or ('contract' in text.lower()):
        return 'Other Organization'
    else:
        return text


# In[53]:


df['Type of ownership']=df['Type of ownership'].apply(company_cleaner)


# In[54]:


plt.figure(figsize=(8,8))
p=sns.countplot(x='Type of ownership',data=df,order=df['Type of ownership'].value_counts().index)
g=p.set_xticklabels(p.get_xticklabels(),rotation=45, horizontalalignment='right')


# In[55]:


df['Industry'].head()


# In[56]:


df['Industry'].nunique()


# In[57]:


df['Industry'].value_counts()[:20]


# In[58]:


df['Industry'] = df['Industry'].apply(lambda x: "others" if x=='-1' else x)


# In[59]:


plt.figure(figsize=(8,8))
p=sns.countplot(x='Industry',data=df,order=df['Industry'].value_counts()[:20].index)
g=p.set_xticklabels(p.get_xticklabels(),rotation=45, horizontalalignment='right')


# In[60]:


df['Sector'].value_counts()


# In[61]:


df['Sector']=df['Sector'].apply(lambda x: 'Others' if x=='-1' else x)


# In[62]:


plt.figure(figsize=(8,8))
p=sns.countplot(x='Sector',data=df,order=df['Sector'].value_counts()[:20].index)
g=p.set_xticklabels(p.get_xticklabels(),rotation=45, horizontalalignment='right')


# In[63]:


df['Revenue'].value_counts()


# In[64]:


df['Revenue']=df['Revenue'].apply(lambda x: "Unknown / Non-Applicable" if x=='-1' else x)


# In[65]:


df['Revenue'].value_counts()


# In[66]:


plt.figure(figsize=(8,8))
p=sns.countplot(x='Revenue',data=df,order=df['Revenue'].value_counts().index)
g=p.set_xticklabels(p.get_xticklabels(),rotation=45, horizontalalignment='right')


# In[67]:


df['Competitors'].head(10)


# In[68]:


df['Competitors']=df['Competitors'].apply(lambda x: len(x.split(',')) if x!=-1 else 0)


# In[69]:


plt.figure(figsize=(8,8))
p=sns.countplot(x='Competitors',data=df,order=df['Competitors'].value_counts().index)


# In[70]:


df.tail()


# In[71]:


df['job in Headquarters'] = df.apply(lambda x : 1 if x['Location']==x['Headquarters'] else 0, axis=1)


# In[72]:


plt.figure(figsize=(6,6))
sns.countplot(x=df['job in Headquarters'],data=df)
plt.xlabel("Is job in HeadQuarters")


# In[73]:


df.drop(labels=['Location'],axis=1,inplace=True)


# In[74]:


df['Job Description'].head(20)


# Creating Columns of Python, SQL, Tableau, Excel

# In[75]:


df['python_job']=df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
df['Excel_job']=df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
df['SQL_job']=df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
df['Tableau_job']=df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)


# In[76]:


plt.figure(figsize=(5,5))
sns.countplot(x='python_job',data=df)
plt.xlabel('Is Python Job?')
plt.title("Count plot for python jobs")


# In[77]:


plt.figure(figsize=(5,5))
sns.countplot(x='Excel_job',data=df)
plt.xlabel('Is Excel Job?')
plt.title("count plot for Excel jobs")


# In[78]:


plt.figure(figsize=(5,5))
sns.countplot(x='SQL_job',data=df)
plt.xlabel('Is SQL Job?')
plt.title("count plot for SQL jobs")


# In[79]:


plt.figure(figsize=(5,5))
sns.countplot(x='Tableau_job',data=df)
plt.xlabel('Is Tableau Job?')
plt.title("count plot for Tableau jobs")


# In[80]:


df.drop(labels=['Job Description'],axis=1,inplace=True)


# In[81]:


df['Sector'].head()


# In[82]:


sectors_list=list(df["Sector"].value_counts()[:10].index)


# In[83]:


def sec_simplifier(text):
    if text not in sectors_list:
        return "Others"
    else:
        return text
df["Sector"]=df["Sector"].apply(sec_simplifier)


# In[84]:


plt.figure(figsize=(8,8))
g=sns.countplot(x='Sector',data=df,order=df['Sector'].value_counts().index)
p=g.set_xticklabels(g.get_xticklabels(),rotation=45,horizontalalignment='right')
plt.title("Final count plot for sectors")


# In[85]:


#job location trimmer
job_location_list=list(df['job_location'].value_counts()[:9].index)
def job_loc_trim(text):
    if text not in job_location_list:
        return 'Others'
    else:
        return text

df['job_location'] = df['job_location'].apply(job_loc_trim)
    
    


# In[86]:


plt.figure(figsize=(8,8))
sns.countplot(x='job_location',data=df,order=df['job_location'].value_counts().index)
plt.title('Final count plot for job location')


# In[87]:


df.drop(labels=['Salary Estimate','Company Name','Headquarters','Industry','job_location'],axis=1,inplace=True)


# In[88]:


df.head()


# In[89]:


df.columns


# In[90]:


df.rename(columns={"Rating":'company_rating','Size':'company_size',"Founded":'company_founded','Type of ownership':'type_of_ownership',
                   'Sector':'sector',"Revenue":'revenue','Competitors':'competitors',"Salary":'salary'},inplace=True)


# In[91]:


# Mapping ranks to 'company_size' column
size_map = {'Unknown': 0, '1 to 50 employees': 1, '51 to 200 employees': 2, '201 to 500 employees': 3,
            '501 to 1000 employees': 4, '1001 to 5000 employees': 5, '5001 to 10000 employees': 6, '10000+ employees': 7}

df['company_size'] = df['company_size'].map(size_map)


# In[92]:


# Mapping ranks to 'revenue	' column
revenue_map = {'Unknown / Non-Applicable': 0, 'Less than $1 million (USD)': 1, '$1 to $5 million (USD)': 2, '$5 to $10 million (USD)': 3,
            '$10 to $25 million (USD)': 4, '$25 to $50 million (USD)': 5, '$50 to $100 million (USD)': 6, '$100 to $500 million (USD)': 7,
            '$500 million to $1 billion (USD)': 8, '$1 to $2 billion (USD)': 9, '$2 to $5 billion (USD)':10, '$5 to $10 billion (USD)':11,
            '$10+ billion (USD)':12}

df['revenue'] = df['revenue'].map(revenue_map)


# In[93]:


# Mapping ranks to 'job_seniority	' column
job_seniority_map = {'other': 0, 'jr': 1, 'sr': 2}

df['job_seniority'] = df['job_seniority'].map(job_seniority_map)


# In[94]:


# Removing 'type_of_ownership' column using get_dummies()
print('Before: {}'.format(df.shape))
df = pd.get_dummies(columns=['type_of_ownership'], data=df, prefix='ownership')
print('After: {}'.format(df.shape))


# In[95]:


# Removing 'sector' column using get_dummies()
print('Before: {}'.format(df.shape))
df = pd.get_dummies(columns=['sector'], data=df)
print('After: {}'.format(df.shape))


# In[96]:


# Removing 'others' column to reduce dimentionality and avoid dummy variable trap
df.drop(labels=['ownership_Other Organization', 'sector_Others'], axis=1, inplace=True)


# In[97]:


df.head()


# In[98]:


X=df.drop('salary',axis=1)


# In[99]:


y=df['salary']


# In[100]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression


# In[101]:


fs=SelectKBest(score_func=mutual_info_regression,k="all")


# In[102]:


fs.fit(X,y)


# In[106]:


df.columns


# In[ ]:


df.head()


# In[107]:


# Removing 'job_title' column using get_dummies()
print('Before: {}'.format(df.shape))
df = pd.get_dummies(columns=['job_title'], data=df)
print('After: {}'.format(df.shape))


# In[111]:


df.drop(labels=[ 'job_title_other'], axis=1, inplace=True)


# In[112]:


X=df.drop('salary',axis=1)


# In[115]:


X.columns


# In[117]:


y.head()


# In[118]:


fs.fit(X,y)


# In[119]:


fs.scores_


# In[120]:


plt.figure(figsize=(8,8))
g=sns.barplot(x=X.columns,y=fs.scores_)
g.set_xticklabels(g.get_xticklabels(),horizontalalignment='right',rotation=45)
plt.xlabel('Feature name')
plt.ylabel("Information gain")


# In[121]:


feature_imp=pd.DataFrame(fs.scores_,columns=['Scores'],index=X.columns)


# In[122]:


top20_features=feature_imp.nlargest(n=20,columns=['Scores'])


# In[123]:


plt.figure(figsize=(8,8))
g=sns.barplot(x=top20_features.index,y=top20_features['Scores'])
g.set_xticklabels(g.get_xticklabels(),rotation=45,horizontalalignment='right')


# In[124]:


corr=X[top20_features.index].corr()


# In[125]:


corr


# In[126]:


plt.figure(figsize=(15,15))
mask=np.triu(np.ones_like(corr,dtype=np.bool))
sns.heatmap(corr,annot=True,mask=mask)

plt.title("Correlation matrix")


# Strong correlation: X>0.7
# 
# moderate correlation: 0.5<X<0.7
# 
# Weak correlation: X<0.5

# #from the above graph it is clear that 
# #(ownership_Public, ownership_Private), (company_size, company_founded), 
# #and(company_founded, revenue) are moderately correlated.
# #Hence, dropping 'Revenue', 'ownership_Public' and 'company_size' features.

# In[127]:


X = X[top20_features.index]
X.drop(labels=['ownership_Public', 'company_size', 'revenue'], axis=1, inplace=True)


# Correlation matrix for top 18 features

# In[130]:


top_18_corr=X.corr()
plt.figure(figsize=(15,15))
mask=np.triu(np.ones_like(top_18_corr,dtype=np.bool))
sns.heatmap(top_18_corr,annot=True,mask=mask)

plt.title("Correlation matrix for top_18 corr")


# In[131]:


X.columns


# In[133]:


X = X[['company_rating', 'company_founded', 'competitors',
       'sector_Health Care', 'sector_Business Services', 'sector_Information Technology',
       'ownership_Private', 'sector_Biotech & Pharmaceuticals',
       'job_title_Data scientist', 'job_title_Data analyst', 'job_seniority', 'job in Headquarters',
       'Excel_job', 'python_job', 'Tableau_job', 'SQL_job',]]


# In[134]:


X.head()


# In[136]:


from sklearn.preprocessing import StandardScaler


# In[140]:


X_prev=X.copy()
sc_rating=StandardScaler()
sc_founded=StandardScaler()


# In[142]:


X['company_rating']=sc_rating.fit_transform(X[['company_rating']])
X['company_founded']=sc_founded.fit_transform(X[['company_founded']])


# In[145]:


plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title("Before feature scaling")
sns.distplot(X_prev['company_rating'])
plt.subplot(1,2,2)
plt.title("After feature scaling")
sns.distplot(X['company_rating'])


# In[146]:


X.head()


# # Model Building and evaluation

# In[147]:


from sklearn.model_selection import cross_val_score


# In[160]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
cv = cross_val_score(lr, X, y, cv=10,scoring='neg_root_mean_squared_error')
print('--- Average NRMSE: {} ---'.format(round(cv.mean(), 3)))
print('Standard Deviation: {}'.format(round(cv.std(), 3)))


# In[161]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
cv = cross_val_score(rf, X, y, cv=10, scoring='neg_root_mean_squared_error')
print('--- Average NRMSE: {} ---'.format(round(cv.mean(), 3)))
print('Standard Deviation: {}'.format(round(cv.std(), 3)))


# In[162]:


from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
cv = cross_val_score(gb, X, y, cv=10, scoring='neg_root_mean_squared_error')
print('--- Average NRMSE: {} ---'.format(round(cv.mean(), 3)))
print('Standard Deviation: {}'.format(round(cv.std(), 3)))


# # We are using random forest regressor

# In[163]:


rf = RandomForestRegressor()
rf.fit(X, y)


# In[170]:


def predict_salary(rating, founded, competitors, sector, ownership, job_title, job_in_headquarters, job_seniority, job_skills):

  prediction_input = list()

  prediction_input.append(sc_rating.transform(np.array(rating).reshape(1, -1)))
  prediction_input.append(sc_founded.transform(np.array(founded).reshape(1, -1)))
  prediction_input.append(competitors)
  

  sector_columns = ['sector_Biotech & Pharmaceuticals', 'sector_Health Care',
                    'sector_Business Services','sector_Information Technology']
  temp = list(map(int, np.zeros(shape=(1, len(sector_columns)))[0]))
  for index in range(0, len(sector_columns)):
    if sector_columns[index] == 'sector_' + sector:
      temp[index] = 1
      break
  prediction_input = prediction_input + temp


  if ownership == 'Private':
    prediction_input.append(1)
  else:
    prediction_input.append(0)
  

  job_title_columns = ['job_title_Data scientist', 'job_title_Data analyst']
  temp = list(map(int, np.zeros(shape=(1, len(job_title_columns)))[0]))
  for index in range(0, len(job_title_columns)):
    if job_title_columns[index] == 'job_title_' + job_title:
      temp[index] = 1
      break
  prediction_input = prediction_input + temp


  prediction_input.append(job_in_headquarters)


  job_seniority_map = {'other': 0, 'jr': 1, 'sr': 2}
  prediction_input.append(job_seniority_map[job_seniority])


  temp = list(map(int, np.zeros(shape=(1, 4))[0]))
  if 'excel' in job_skills:
    temp[0] = 1
  if 'python' in job_skills:
    temp[1] = 1
  if 'tableau' in job_skills:
    temp[2] = 1
  if 'sql' in job_skills:
    temp[3] = 1
  prediction_input = prediction_input + temp


  return rf.predict([prediction_input])[0]


# In[171]:


# Prediction 1
# Input sequence: 'company_rating', 'company_founded', 'competitors_count',
#                 'company_sector', 'company_ownership', 'job_title', 'job_in_headquarters',
#                 'job_seniority', 'job_skills'

salary = predict_salary(4.5, 1969, 3, 'Information Technology', 'Private', 'data scientist', 1, 'sr', ['python', 'sql', 'tableau'])
print('Estimated salary (range): {}(USD) to {}(USD) per annum.'.format(int(salary*1000)-9000, int(salary*1000)+9000))


# In[166]:


# Prediction 2
# Input sequence: 'company_rating', 'company_founded', 'competitors_count',
#                 'company_sector', 'company_ownership', 'job_title', 'job_in_headquarters',
#                 'job_seniority', 'job_skills'

salary = predict_salary(3.0, 2000, 1, 'Health Care', 'Public', 'data analyst', 0, 'jr', ['python', 'tableau'])
print('Estimated salary (range): {}(USD) to {}(USD) per annum.'.format(int(salary*1000)-9000, int(salary*1000)+9000))


# In[ ]:




