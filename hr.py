import pandas as pd   
import numpy as np  
import matplotlib.pyplot as plt                                     # Importing pyplot interface using matplotlib
import seaborn as sns                                               # Importin seaborm library for interactive visualization
import plotly.graph_objs as go                                      # Importing plotly for interactive visualizations

from sklearn.preprocessing import StandardScaler                   # Importing to scale the features in the dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split                # To properly split the dataset into train and test sets
from sklearn.linear_model import LogisticRegression                   # To create a linear regression model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier                  # To create a random forest regressor model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB                          # To create a naive bayes model using algorithm
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics                                         # Importing to evaluate the model used for regression
from sklearn.decomposition import PCA                               # Importing to create an instance of PCA model
#-------------------------------------------------------------------------------------------------------------------------------
from random import randint                                          # Importing to generate random integers
#-------------------------------------------------------------------------------------------------------------------------------
import time                                                         # For time functionality
import warnings                                                     # Importing warning to disable runtime warnings
warnings.filterwarnings("ignore")                                   # Warnings will appear only once

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import mysql.connector
import xgboost
from xgboost import XGBClassifier

def conn_db():
              mydb=mysql.connector.connect(host='cpanel.insaid.co',
                        database='Capstone2',
                        user='student',
                        password='student')
              return mydb

def datagather():

    mydb=conn_db()

    department_data=pd.read_sql("SELECT * FROM department_data ",mydb)
    employee_details_data=pd.read_sql("SELECT * FROM employee_details_data",mydb)
    employee_data=pd.read_sql("SELECT * FROM employee_data",mydb)

    mydb.close()

    return department_data,employee_details_data, employee_data


def data_Pre_Processing(department_data, employee_details_data, employee_data):

   
   
    not_null_emp_data = employee_data.isna().sum()
    duplicated_emp_data = employee_data.duplicated().sum()
    employee_data.drop_duplicates(inplace=True)

    index_names = employee_data[employee_data['employee_id']==0].index
    employee_data.drop(index_names, inplace = True)
    employee_data.shape

    employee_data["filed_complaint"].isna().sum()
    print("Filed Complaint",employee_data["filed_complaint"].isna().sum() )

    employee_data["filed_complaint"].fillna(0,inplace=True)
    employee_data["filed_complaint"].unique()
    employee_data["filed_complaint"]=employee_data["filed_complaint"].astype('int64')


    employee_data["recently_promoted"].unique()
    employee_data["recently_promoted"].fillna(0,inplace=True)
    employee_data["recently_promoted"]=employee_data["recently_promoted"].astype('int64')

    employee_data["department"] = employee_data["department"].replace({"-IT":"D00-IT"})
    employee_data["department"].fillna("D00-Missing",inplace = True)
  
    employee_data.drop(employee_data[employee_data['department']=="D00-TP"].index, inplace = True)
    employee_data.last_evaluation= employee_data.groupby('department').last_evaluation.apply(lambda x: x.fillna(x.median()))
    df_final= pd.merge(employee_data, employee_details_data, on='employee_id', how='inner')

    return df_final

def data_feature_engineering(df_final):

    df_final.info()
    data_categorical = df_final[['department','status','salary']]
    data_categorical.head()
    data_categorical = data_categorical.apply(LabelEncoder().fit_transform)
    data_categorical.head()

    data_numerical = df_final[[ 'avg_monthly_hrs', 'filed_complaint', 'last_evaluation','n_projects', 'recently_promoted', 'satisfaction','tenure', 'employee_id']]
    data_numerical.head()

    data_model = pd.concat([data_categorical,data_numerical], axis=1)
    data_model.head()

    return data_model

def internal_tts(data_model):

    y = data_model[['status']]
    x2 = data_model.drop('status',axis=1)
    x2.head()
    x = x2.drop('employee_id',axis=1)
    x.head()

    # Splitting data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Display the shape of training and testing data
    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)
    print('x_test shape: ', x_test.shape)
    print('y_test shape: ', y_test.shape)

    return x_train, x_test, y_train, y_test


def model_building(x_train, x_test, y_train, y_test):

    clfs = [LogisticRegression(solver='liblinear'),DecisionTreeClassifier(random_state = 0), RandomForestClassifier(n_estimators = 100, random_state = 0, n_jobs = -1)
       ,KNeighborsClassifier(n_neighbors = 6),GaussianNB(), GradientBoostingClassifier(random_state = 0),XGBClassifier(n_estimators=64, random_state=42)]

    for clf in clfs:    

  # Extracting model name
        model_name = type(clf).__name__

  # Calculate start time
        start_time = time.time()

  # Train the model
        clf.fit(x_train, y_train)

  # Make predictions on the trained model
        predictions = clf.predict(x_test)


        ACCURACY_SCORE = accuracy_score(y_test,predictions)
        CLASSIFICATION_REPORT=classification_report(y_test,predictions)
        F1_SCORE=f1_score(y_test,predictions)

        # Calculate evaluated time
        elapsed_time = (time.time() - start_time)

        # Display the metrics and time took to develop the model
        print('Performance Metrics of', model_name, ':')
        print('[ACCURACY SCORE]:', ACCURACY_SCORE, '[Processing Time]:', elapsed_time, 'seconds')
        print('[CLASSIFICATION REPORT]:', CLASSIFICATION_REPORT)

        print('[F1_SCORE]:',F1_SCORE)
        print('----------------------------------------\n')



    return ACCURACY_SCORE, CLASSIFICATION_REPORT, F1_SCORE

dept_data = datagather();
department_data = dept_data[0]
employee_details_data = dept_data[1]
employee_data = dept_data[2]


df_final = data_Pre_Processing(department_data,employee_details_data, employee_data )

data_model = data_feature_engineering(df_final)

tts = internal_tts(data_model)

x_train = tts[0]
x_test = tts[1]
y_train = tts[2]
y_test = tts[3]

mod = model_building(x_train,x_test,y_train, y_test )

accuracy_sc = mod[0]
class_rep = mod[1]
fi_sc = mod[2]