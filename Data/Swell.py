import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, decomposition, metrics, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from xgboost import XGBClassifier
import pickle

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\m45m4\Documents\Computing project\The SWELL Knowledge Work Dataset\D - Physiology features (HR_HRV_SCL - final).csv")

#removing unneeded columns
del df ['Unnamed: 7']
del df ['Unnamed: 8']
del df ['Unnamed: 9']
del df ['Unnamed: 10']
del df ['Unnamed: 11']
del df ['PP']
del df ['C']
del df ['RMSSD']
del df ['SCL']
del df ['timestamp']

#remove rows with missing values
df.dropna(axis =0, how = 'any', subset= None, inplace=True)

#remove any rows that have 999 in any column (this featured in the body sensor column)
df = df[(df.HR != 999)]


#Create new column 'Stressed' with the condition of assigning binary to condition
df['Stressed'] = [0 if x == 'R' else 1 for x in df['Condition']]

print (df)

#Visualise data for Heart rate levels against stress
#df.boxplot(column="HR", by='Stressed')
#plt.xlabel('Stressed')
#plt.ylabel('Heart Rate')
#plt.title("Heart Rate correlation to stress")
#plt.suptitle('')
#plt.show()

#Visualise data for RMSSD against stress
#df.boxplot(column="RMSSD", by='Stressed')
#plt.xlabel('Stressed')
#plt.ylabel('RMSSD')
#plt.title("RMSSD correlation to stress")
#plt.suptitle('')
#plt.show()

#Participant 1 data only
#pd.set_option('display.max_rows',None)
#rslt_df = df[(df['PP'] == 'PP1')]
#print(rslt_df)

#rslt_df.boxplot(column="HR", by='Stressed')
#plt.xlabel('Stressed')
#plt.ylabel('Heart Rate (BPM)')
#plt.title("Heart Rate correlation to stress for Participant 1")
#plt.suptitle('')
#plt.show()

# defining x as independent variables, a y as dependent
x = df[['HR']]
y = df['Stressed']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#traning the model
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

#evaluate model performance
y_pred = dt_clf_gini.predict(X_test)
print ("Accuracy:", accuracy_score(y_test, y_pred))

#average=weighted means to combine precision values from multiple classes
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"Precision: {precision}")
print('Precision2: %.3f' % precision_score(y_test, y_pred, average='weighted', zero_division=0))

recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
print(f"Recall: {recall}")
print('Recall2: %.3f' % recall_score(y_test, y_pred, average='weighted', zero_division=0))

#F1 Score
print('F1 Score: %.3f' % f1_score(y_test, y_pred, average='weighted', zero_division=0))

#save the model
with open ("Swell.pkl", "wb") as model_file:
    pickle.dump(dt_clf_gini, model_file)

#how much data is being used for training and how much for testing 
#print (X_train.shape, X_test.shape)

#print( X_train,'\n', X_test)
#print( y_train,'\n', y_test)

#normalising data for stabilisation
#norm = MinMaxScaler()
#X_train= norm.fit_transform(X_train)
#X_test = norm.transform(X_test)

#training
#models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

#models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

#for i in range(3):
 #   models[i].fit(X_train, y_train)

  #  print(f'{models[i]} : ')
   # print('Training Accuracy : ', metrics.roc_auc_score(y_train, models[i].predict(X_train)))
    #print('Validation Accuracy : ', metrics.roc_auc_score(
     #   y_test, models[i].predict(X_test)))
    #print()

#model evaluation
#print(metrics.classification_report(y_test, models[1].predict(X_test)))


#saving trained model
#with open("model.pkl","wb") as model_file:
 #   pickle.dump(dt_clf_gini)