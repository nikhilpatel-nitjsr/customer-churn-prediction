#!/usr/bin/env python
# coding: utf-8

# # TELECOM CUSTOMER CHURN PREDICTION

# * Customer churn is defined as when customers or subscribers discontinue doing business with a firm or service.
# * Individualized customer retention is tough because most firms have a large number of customers and can't afford to devote much time to each of them. The costs would be too great, outweighing the additional revenue. However, if a corporation could forecast which customers are likely to leave ahead of time, it could focus customer retention efforts only on these "high risk" clients. The ultimate goal is to expand its coverage area and retrieve more customers loyalty. The core to succeed in this market lies in the customer itself
# * Customer churn is a critical metric because it is much less expensive to retain existing customers than it is to acquire new customers.
# * To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.

# ### Objectives

# #### I will explore the data and try to answer some questions like:
# * What's the % of Churn Customers and customers that keep in with the active services?
# * Is there any patterns in Churn Customers based on the gender?
# * Is there any patterns/preference in Churn Customers based on the type of service provided?
# * What's the most profitable service types?
# * Which features and services are most profitable?
# * Many more questions that will arise during the analysis

# #### A brief explanation of this dataset:
# ##### This dataset is IBM Sample Data Sets that I founded at Kaggle.
# * Each row represents a customer; each column contains the customer’s attributes described in the column Metadata.
# * The data set includes information about:
# * Customers who left within the last month — the column is called Churn.
# * Services that each customer has signed up for — phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# * Customer account information — how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# * Demographic info about customers — gender, age range, and if they have partners and dependents

# #### Steps followed during whole analysis:-
# * Step 1: Gather the data
# * Step 2: Assess and clean the data
# * Step 3: Conduct exploratory data analysis to answer the questions & create visualizations (Final visualization code)
# * Step 4: Train,adjust and evaluate the model.
# * Step 5: Summaries

# #### Importing the libraries for data loading,visualizition.

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[84]:


df = pd.read_csv('telecom_customer_churn.csv')
df.head()


# In[85]:


df.shape


# In[86]:


df.info()


# ### Data Manipulation

# In[87]:


df.dtypes


# So,almost all features are of object type.
# 
# Since Total Charge column contain numeric data but its type is object so let fix it.

# In[88]:


df.columns.values


# In[89]:


df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors='coerce')


# Lets check for null values in our data

# In[90]:


df.isnull().sum()


# * Only Total charge columns have null values
# * So, we have 2 options we can drop the rows which contains the null values or we can fill it various imputation technique.

# ##### Lets drop the all null values

# ### Visualize missing values

# In[91]:


df = df.dropna()
df.isnull().sum()


# there is no missing data.

# In[92]:


df.describe()


# ## Exploratory Data Analysis

# Lets look the balance of class labels

# In[93]:


df["Churn"].value_counts()


# In[94]:


sns.countplot(data=df, x="Churn",hue="gender")
plt.title("Gender Vs Churn");


# From the above graph, we can see that gender is not a contributing factor for customer churn in this data set as the numbers of both the genders, that have or haven’t churned, are almost the same.

# In[95]:


sns.countplot(data=df,x="Churn",hue="SeniorCitizen")
plt.title("Senior Citizen Vs Churn ");


# the rate of churning is high among the non senior citizen

# #### Lets look at the distribution of Churn with Total Charges

# In[96]:


plt.figure(figsize=(15,10));
sns.boxplot(data=df,x="Churn",y="TotalCharges",hue="gender");


# there is similar distribution of male and feamle and the pepole who have higher total charges are less likely to churn.

# In[97]:


plt.figure(figsize=(15,10));
sns.boxplot(data=df,x="Contract",y="TotalCharges",hue="Churn")
plt.title("Distribution of Total Charge vs Contract type seperated by Churn.")
plt.xticks(rotation=45)


# So,here pepole who have contract for more time and who paid more are likely to churn. We can reedem them by providing some offers after 1 year or 2 year completion

# In[98]:


plt.figure(figsize=(15,5));
sns.countplot(x='Churn',data=df, hue='InternetService')


# We can see that people using Fiber-optic services have a higher churn percentage. This shows that the company needs to improve their Fiber-optic service

# In[99]:


plt.figure(figsize=(15,5));
sns.countplot(x='TechSupport',data=df, hue='Churn',palette='viridis')


# Those customers who don’t have tech support have churned more, which is pretty self-explanatory. This also highlights the fact that the tech support provided by the company is up to the mark.

# ### Lets look at the correlation of features with churn.

# Since most features are categorical feature so let first create dummy varibale of some important feature and than apply corr( ) function to get he correlation with respect to yes churn.

# In[100]:


corr_df  = pd.get_dummies(df[['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod','Churn']]).corr()


# In[101]:


corr_df.head()


# In[102]:


corr_df["Churn_Yes"].sort_values().iloc[1:-1]


# Also ploting the correlation

# In[103]:


plt.figure(figsize=(20,8));

sns.barplot(data=corr_df["Churn_Yes"].sort_values().iloc[1:-1],x=corr_df["Churn_Yes"].sort_values().iloc[1:-1].index,y=corr_df["Churn_Yes"].sort_values().iloc[1:-1].values)
plt.xticks(rotation=45);
plt.title("Feature Correlation to Yes Churn");


# ## Churn Analysis

# This section focuses on segementing customers based on their tenure, creating "cohorts", allowing us to examine differences between customer cohort segments.

# Lets find the different types of contract that is available.

# In[104]:


df['Contract'].unique()


# In[105]:


plt.figure(figsize=(10,4),dpi=200)
sns.histplot(data=df,x='tenure',bins=60)
plt.title("Distribution of Tenure");


# * more people have 1 or 2 month tenure and then more number of people is of more than 2 years contract type.
# 
# * So, here we can conclude that either customer need the service for short duration of time or a very long duration of time.

# In[106]:


plt.figure(figsize=(15,8));

ax = sns.histplot(x = 'tenure', hue = 'Churn', data = df, multiple='dodge')
ax.set(xlabel="Tenure in Months", ylabel = "Count");


# The churn amount is higher in the initial 5 months, which is usually the time when the new customers try out the service and decide whether to continue or cancel. This pretty much can be attributed to the uncertainty in the customer’s mind.

# ### Ploting Total Charge vs Monthly Charges with separated by churn.

# In[107]:


plt.figure(figsize=(10,4),dpi=200)
sns.scatterplot(data=df,x='MonthlyCharges',y='TotalCharges',hue='Churn', linewidth=0.5,alpha=0.5,palette='Dark2')


# Here people who have more monthly charge instead of having lower total charge is more likely to churn.

# #### Lets look the churn rate of cusmtomer based on tenure month.

# In[108]:


no_churn = df.groupby(['Churn','tenure']).count().transpose()['No']
yes_churn = df.groupby(['Churn','tenure']).count().transpose()['Yes']
no_churn


# In[109]:


churn_rate = 100 * yes_churn / (no_churn+yes_churn)


# In[110]:


churn_rate.transpose()['customerID']


# #### ploting the same

# In[111]:


plt.figure(figsize=(10,4),dpi=200)
churn_rate.iloc[0].plot()
plt.ylabel('Churn Percentage');


# churn rate gradually decreases with the increase of tenure

# ### Predictive Modeling

# Separating our data into features and labels

# In[112]:


X = df.drop(['Churn','customerID'],axis=1)
X = pd.get_dummies(X,drop_first=True)


# In[113]:


y = df['Churn']


# A train test split, holding out 20% of the data for testing. We'll use a random_state of 101 to comapre different types of model.

# In[114]:


from sklearn.model_selection import train_test_split


# In[115]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# ### Normalizing our feature since we are also going to use distance based modal.

# In[116]:


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report,plot_confusion_matrix,accuracy_score


# In[117]:


scaler = StandardScaler()


# In[118]:


scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)


# In[119]:


from sklearn.metrics import accuracy_score,plot_confusion_matrix,classification_report


# ## Logistic Regression

# * Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.
# * Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.
# * Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems.
# * In Logistic regression, instead of fitting a regression line, we fit an "S" shaped logistic function, which predicts two maximum values (0 or 1).
# * The curve from the logistic function indicates the likelihood of something such as whether the cells are cancerous or not, a mouse is obese or not based on its weight, etc.
# * Logistic Regression is a significant machine learning algorithm because it has the ability to provide probabilities and classify new data using continuous and discrete datasets.
# * Logistic Regression can be used to classify the observations using different types of data and can easily determine the most effective variables used for the classification.
# * Logistic regression uses the concept of predictive modeling as regression; therefore, it is called logistic regression, but is used to classify samples; Therefore, it falls under the classification algorithm.
# 

# #### Sigmoid Function

# * The sigmoid function is a mathematical function used to map the predicted values to probabilities.
# * It maps any real value into another value within a range of 0 and 1.
# * The value of the logistic regression must be between 0 and 1, which cannot go beyond this limit, so it forms a curve like the "S" form. The S-form curve is called the Sigmoid function or the logistic function.
# * In logistic regression, we use the concept of the threshold value, which defines the probability of either 0 or 1. Such as values above the threshold value tends to 1, and a value below the threshold values tends to 0.

# #### Steps in Logistic Regression: 
# * To implement the Logistic Regression using Python, we will use the same steps as we have done in previous topics of Regression. Below are the steps:
# 
# * Data Pre-processing step
# * Fitting Logistic Regression to the Training set
# * Predicting the test result
# * Test accuracy of the result(Creation of Confusion matrix)
# * Visualizing the test set result.

# In[120]:


from sklearn.linear_model import LogisticRegression 
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
prediction_logreg=logreg.predict(X_test)
print(accuracy_score(y_test,prediction_logreg))


# In[161]:


confusion_matrix(y_test,y_pred)


# In[162]:


print(classification_report(y_test,y_pred))


# In[163]:


plot_confusion_matrix(logreg,scaled_X_test,y_test)


# ## Decision Tree Classification Algorithm

# * Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.
# * In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.
# * The decisions or the test are performed on the basis of features of the given dataset.
# * It is a graphical representation for getting all the possible solutions to a problem/decision based on given conditions.
# * It is called a decision tree because, similar to a tree, it starts with the root node, which expands on further branches and constructs a tree-like structure.
# * In order to build a tree, we use the CART algorithm, which stands for Classification and Regression Tree algorithm.
# * A decision tree simply asks a question, and based on the answer (Yes/No), it further split the tree into subtrees.

# #### How does the Decision Tree algorithm Work?
# * Step-1: Begin the tree with the root node, says S, which contains the complete dataset.
# * Step-2: Find the best attribute in the dataset using Attribute Selection Measure (ASM).
# * Step-3: Divide the S into subsets that contains possible values for the best attributes.
# * Step-4: Generate the decision tree node, which contains the best attribute.
# * Step-5: Recursively make new decision trees using the subsets of the dataset created in step -3. Continue this process until a stage is reached where you cannot further classify the nodes and called the final node as a leaf node.

# ###### Let import DecisionTreeClassifier from sklearn and tarin it than evaluate it.

# In[121]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train,y_train)


# In[122]:


preds = dt.predict(X_test)


# In[123]:


print(classification_report(y_test,preds))


# In[124]:


plot_confusion_matrix(dt,X_test,y_test)


# In[125]:


param_grid_dt = {"criterion" : ["gini", "entropy"],
             "max_depth":[1,2,6,10,14,20],
             }


# In[126]:


from sklearn.model_selection import GridSearchCV
grid_dt = GridSearchCV(estimator=dt,param_grid=param_grid_dt)


# In[127]:


grid_dt.fit(X_train,y_train)


# In[128]:


grid_dt.best_params_


# In[129]:


grid_dt_pre=grid_dt.predict(X_test)
confusion_matrix(y_test,grid_dt_pre)


# In[130]:


print(classification_report(y_test,grid_dt_pre))


# In[131]:


plot_confusion_matrix(grid_dt,X_test,y_test)


# In[132]:


from sklearn.tree import plot_tree


# In[133]:


plt.figure(figsize=(12,8),dpi=500)
plot_tree(dt,filled=True,feature_names=X.columns,max_depth=2);


# ## Random Forest Algorithm

# * Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.
# * As the name suggests, "Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset." Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.
# * The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.

# #### How does Random Forest algorithm work?
# 
# 
# * Random Forest works in two-phase first is to create the random forest by combining N decision tree, and second is to make predictions for each tree created in the first phase.
# ###### The Working process can be explained in the below steps and diagram:
# * Step-1: Select random K data points from the training set.
# * Step-2: Build the decision trees associated with the selected data points (Subsets).
# * Step-3: Choose the number N for decision trees that you want to build.
# * Step-4: Repeat Step 1 & 2.
# * Step-5: For new data points, find the predictions of each decision tree, and assign the new data points to the category that wins the majority votes.

# ### Importing, Training and Evaluating RandomForestClassifier.

# In[134]:


from sklearn.ensemble import RandomForestClassifier


# In[135]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(scaled_X_train,y_train)
preds = rf.predict(scaled_X_test)
print(classification_report(y_test,preds))


# In[136]:


plot_confusion_matrix(rf,X_test,y_test)


# In[137]:


param_grid_rf = {
             "n_estimators":[100,200,500,1000],
'criterion': ['gini', 'entropy'],
                         'max_depth': [1, 2, 6, 10, 14, 20]}
from sklearn.model_selection import GridSearchCV


# In[138]:


grid_rf =GridSearchCV(estimator=RandomForestClassifier(),param_grid=param_grid_rf)


# In[139]:


grid_rf.fit(X_train,y_train)


# In[140]:


grid_rf.best_params_


# In[141]:


grid_rf_pred=grid_rf.predict(X_test)


# In[142]:


print(classification_report(y_test,grid_rf_pred))


# In[143]:


plot_confusion_matrix(grid_rf,X_test,y_test)


# ## KNN model and its evalution:-

# * K-Nearest Neighbour is one of the simplest Machine Learning algorithms based on Supervised Learning technique.
# * K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories.
# * K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm.
# * K-NN algorithm can be used for Regression as well as for Classification but mostly it is used for the Classification problems.
# * K-NN is a non-parametric algorithm, which means it does not make any assumption on underlying data.
# * It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.
# * KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

# ### How does K-NN work?

# ##### The K-NN working can be explained on the basis of the below algorithm:
# * Step-1: Select the number K of the neighbors
# * Step-2: Calculate the Euclidean distance of K number of neighbors
# * Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
# * Step-4: Among these k neighbors, count the number of the data points in each category.
# * Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
# * Step-6: Our model is ready.

# ### Importing KNN classifier model from Sklearn

# In[144]:


from sklearn.neighbors import KNeighborsClassifier


# In[145]:


knn_model = KNeighborsClassifier(n_neighbors=1)


# In[146]:


knn_model.fit(scaled_X_train,y_train)


# In[147]:


y_pred = knn_model.predict(scaled_X_test)


# In[148]:


accuracy_score(y_test,y_pred)


# In[149]:


confusion_matrix(y_test,y_pred)


# In[150]:


print(classification_report(y_test,y_pred))


# In[151]:


plot_confusion_matrix(knn_model,scaled_X_test,y_test)


# So, KNN gives a accuracy of 0.72.

# ### Elbow method to select best K value.

# In[152]:


test_error_rates = []


for k in range(1,40):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(scaled_X_train,y_train) 
   
    y_pred_test = knn_model.predict(scaled_X_test)
    
    test_error = 1 - accuracy_score(y_test,y_pred_test)
    test_error_rates.append(test_error)


# In[153]:


plt.figure(figsize=(10,6),dpi=200)
plt.plot(range(1,40),test_error_rates,label='Test Error')
plt.legend()
plt.ylabel('Error Rate')
plt.xlabel("K Value")


# So, k value of 12 will be optimum.

# In[154]:


knn_model = KNeighborsClassifier(n_neighbors=12)


# In[155]:


knn_model.fit(scaled_X_train,y_train)


# In[156]:


y_pred = knn_model.predict(scaled_X_test)


# In[157]:


accuracy_score(y_test,y_pred)


# In[158]:


confusion_matrix(y_test,y_pred)


# In[159]:


print(classification_report(y_test,y_pred))


# In[160]:


plot_confusion_matrix(knn_model,scaled_X_test,y_test)


# After choosing a best value of k we have now accuracy of 0.79

# #### FINDING AND SUMMARY

# So, our main focous is that our model does not predict a customer who is going to churn as not going to churn.

# So, now at looking the confusion matrix of various model .

# In[165]:


plot_confusion_matrix(logreg,scaled_X_test,y_test)
plt.title("LOGISTIC REGRESSION");


# In[166]:


plot_confusion_matrix(knn_model,scaled_X_test,y_test);
plt.title("KNN_MODEL");


# In[167]:


plot_confusion_matrix(dt,X_test,y_test)
plt.title("DECISION TREE MODEL");


# In[168]:


plot_confusion_matrix(grid_rf,X_test,y_test)
plt.title("RANDOM FOREST MODEL");


# Clearly the logistic regression model are the best model here as they are the one who minimum miss classify the true_yes vs predicted_no.

# In[ ]:




