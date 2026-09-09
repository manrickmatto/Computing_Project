import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
import lightgbm as lgb


#s2 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S2\S2.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df1 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df1 = pd.DataFrame(df1)
df1 = df1[df1 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df)

#remove unneeded data label 0
rows = df1[df1["emotion_label"] == 0].index
df1.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 5
rows = df1[df1["emotion_label"] == 5].index
df1.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 6 
rows = df1[df1["emotion_label"] == 6].index
df1.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 7
rows = df1[df1["emotion_label"] == 7].index
df1.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df1)

#check for null values
df1 = pd.DataFrame(df1, columns=['ecg','emotion_label'])

check_nan = df1 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan
###################################################################################################################

#s3 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S3\S3.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df3 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df3 = pd.DataFrame(df3)
df3 = df3[df3 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df)

#remove unneeded data label 0
rows = df3[df3["emotion_label"] == 0].index
df3.drop(rows, inplace=True)
#print(df)


#remove unneeded data label 5
rows = df3[df3["emotion_label"] == 5].index
df3.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 6 
rows = df3[df3["emotion_label"] == 6].index
df3.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 7
rows = df3[df3["emotion_label"] == 7].index
df3.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df3)

#check for null values
df3 = pd.DataFrame(df3, columns=['ecg','emotion_label'])

check_nan = df3 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#############################################################################################
#S4PKL 
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S4\S4.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df4 = df_ecg.join(df_stress_label_chest)
#print (df4)

#remove negative values
df4 = pd.DataFrame(df4)
df4 = df4[df4 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df)

#remove unneeded data label 0
rows = df4[df4["emotion_label"] == 0].index
df4.drop(rows, inplace=True)
#print(df4)

#remove unneeded data label 5
rows = df4[df4["emotion_label"] == 5].index
df4.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 6 
rows = df4[df4["emotion_label"] == 6].index
df4.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 7
rows = df4[df4["emotion_label"] == 7].index
df4.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df4)

#check for null values
df4 = pd.DataFrame(df4, columns=['ecg','emotion_label'])

check_nan = df4 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan
###################################################################################################

#S5pkl 
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S5\S5.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df5 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df5 = pd.DataFrame(df5)
df5 = df5[df5 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df)

#remove unneeded data label 0
rows = df5[df5["emotion_label"] == 0].index
df5.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 5
rows = df5[df5["emotion_label"] == 5].index
df5.drop(rows, inplace=True)
#print(df5)

#remove unneeded data label 6 
rows = df5[df5["emotion_label"] == 6].index
df5.drop(rows, inplace=True)
#print(df5)

#remove unneeded data label 7
rows = df5[df5["emotion_label"] == 7].index
df5.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df5)

#check for null values
df5 = pd.DataFrame(df5, columns=['ecg','emotion_label'])

check_nan = df5 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

###########################################################################################################################
#S6 pkl data

file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S6\S6.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df6 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df6 = pd.DataFrame(df6)
df6 = df6[df6 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df6)

#remove unneeded data label 0
rows = df6[df6["emotion_label"] == 0].index
df6.drop(rows, inplace=True)
#print(df6)

#remove unneeded data label 5
rows = df6[df6["emotion_label"] == 5].index
df6.drop(rows, inplace=True)
#print(df6)

#remove unneeded data label 6 
rows = df6[df6["emotion_label"] == 6].index
df6.drop(rows, inplace=True)
#print(df6)

#remove unneeded data label 7
rows = df6[df6["emotion_label"] == 7].index
df6.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df6)

#check for null values
df6 = pd.DataFrame(df6, columns=['ecg','emotion_label'])

check_nan = df6 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#S7pkl 
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S7\S7.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df7 = df_ecg.join(df_stress_label_chest)
#print (df7)

#remove negative values
df7 = pd.DataFrame(df7)
df7 = df7[df7 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df7)

#remove unneeded data label 0
rows = df7[df7["emotion_label"] == 0].index
df7.drop(rows, inplace=True)
#print(df7)

#remove unneeded data label 5
rows = df7[df7["emotion_label"] == 5].index
df7.drop(rows, inplace=True)
#print(df7)

#remove unneeded data label 6 
rows = df7[df7["emotion_label"] == 6].index
df7.drop(rows, inplace=True)
#print(df7)

#remove unneeded data label 7
rows = df7[df7["emotion_label"] == 7].index
df7.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df7)

#check for null values
df7 = pd.DataFrame(df7, columns=['ecg','emotion_label'])

check_nan = df7 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan
#####################################################################################################################
#S8pkl 
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S8\S8.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df8 = df_ecg.join(df_stress_label_chest)
#print (df8)

#remove negative values
df8 = pd.DataFrame(df8)
df8 = df8[df8 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df8)

#remove unneeded data label 0
rows = df8[df8["emotion_label"] == 0].index
df8.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 5
rows = df8[df8["emotion_label"] == 5].index
df8.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 6 
rows = df8[df8["emotion_label"] == 6].index
df8.drop(rows, inplace=True)
#print(df8)

#remove unneeded data label 7
rows = df8[df8["emotion_label"] == 7].index
df8.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df8)

#check for null values
df8 = pd.DataFrame(df8, columns=['ecg','emotion_label'])
check_nan = df8 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s9 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S9\S9.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df9 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df9 = pd.DataFrame(df9)
df9 = df9[df9 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df9)

#remove unneeded data label 0
rows = df9[df9["emotion_label"] == 0].index
df9.drop(rows, inplace=True)
#print(df9)

#remove unneeded data label 5
rows = df9[df9["emotion_label"] == 5].index
df9.drop(rows, inplace=True)
#print(df9)

#remove unneeded data label 6 
rows = df9[df9["emotion_label"] == 6].index
df9.drop(rows, inplace=True)
#print(df9)

#remove unneeded data label 7
rows = df9[df9["emotion_label"] == 7].index
df9.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df9)

#check for null values
df9 = pd.DataFrame(df9, columns=['ecg','emotion_label'])

check_nan = df9 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s10 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S10\S10.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df10 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df10 = pd.DataFrame(df10)
df10 = df10[df10 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df)

#remove unneeded data label 0
rows = df10[df10["emotion_label"] == 0].index
df10.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 5
rows = df10[df10["emotion_label"] == 5].index
df10.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 6 
rows = df10[df10["emotion_label"] == 6].index
df10.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 7
rows = df10[df10["emotion_label"] == 7].index
df10.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df10)

#check for null values
df10 = pd.DataFrame(df10, columns=['ecg','emotion_label'])

check_nan = df10 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s11 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S11\S11.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df11 = df_ecg.join(df_stress_label_chest)
#print (df11)

#remove negative values
df11 = pd.DataFrame(df11)
df11 = df11[df11 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df11)

#remove unneeded data label 0
rows = df11[df11["emotion_label"] == 0].index
df11.drop(rows, inplace=True)
#print(df11)

#remove unneeded data label 5
rows = df11[df11["emotion_label"] == 5].index
df11.drop(rows, inplace=True)
#print(df11)

#remove unneeded data label 6 
rows = df11[df11["emotion_label"] == 6].index
df11.drop(rows, inplace=True)
#print(df11)

#remove unneeded data label 7
rows = df11[df11["emotion_label"] == 7].index
df11.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df11)

#check for null values
df11 = pd.DataFrame(df11, columns=['ecg','emotion_label'])

check_nan = df11 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s13 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S13\S13.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df13 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df13 = pd.DataFrame(df13)
df13 = df13[df13 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df13)

#remove unneeded data label 0
rows = df13[df13["emotion_label"] == 0].index
df13.drop(rows, inplace=True)
#print(df13)

#remove unneeded data label 5
rows = df13[df13["emotion_label"] == 5].index
df13.drop(rows, inplace=True)
#print(df13)

#remove unneeded data label 6 
rows = df13[df13["emotion_label"] == 6].index
df13.drop(rows, inplace=True)
#print(df13)

#remove unneeded data label 7
rows = df13[df13["emotion_label"] == 7].index
df13.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df13)

#check for null values
df13 = pd.DataFrame(df13, columns=['ecg','emotion_label'])

check_nan = df13 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s14 pkl file
file_path = r"c:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S14\S14.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df14 = df_ecg.join(df_stress_label_chest)
#print (df14)

#remove negative values
df14 = pd.DataFrame(df14)
df14 = df14[df14 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df14)

#remove unneeded data label 0
rows = df14[df14["emotion_label"] == 0].index
df14.drop(rows, inplace=True)
#print(df14)

#remove unneeded data label 5
rows = df14[df14["emotion_label"] == 5].index
df14.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 6 
rows = df14[df14["emotion_label"] == 6].index
df14.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 7
rows = df14[df14["emotion_label"] == 7].index
df14.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df14)

#check for null values
df14 = pd.DataFrame(df14, columns=['ecg','emotion_label'])

check_nan = df14 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s15 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S15\S15.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df15 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df15 = pd.DataFrame(df15)
df15 = df15[df15 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df)

#remove unneeded data label 0
rows = df15[df15["emotion_label"] == 0].index
df15.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 5
rows = df15[df15["emotion_label"] == 5].index
df15.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 6 
rows = df15[df15["emotion_label"] == 6].index
df15.drop(rows, inplace=True)
#print(df)

#remove unneeded data label 7
rows = df15[df15["emotion_label"] == 7].index
df15.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df15)

#check for null values
df15 = pd.DataFrame(df15, columns=['ecg','emotion_label'])

check_nan = df15 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s16 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S16\S16.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df16 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df16 = pd.DataFrame(df16)
df16 = df16[df16 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df16)

#remove unneeded data label 0
rows = df16[df16["emotion_label"] == 0].index
df16.drop(rows, inplace=True)
#print(df16)

#remove unneeded data label 5
rows = df16[df16["emotion_label"] == 5].index
df16.drop(rows, inplace=True)
#print(df16)

#remove unneeded data label 6 
rows = df16[df16["emotion_label"] == 6].index
df16.drop(rows, inplace=True)
#print(df16)

#remove unneeded data label 7
rows = df16[df16["emotion_label"] == 7].index
df16.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df16)

#check for null values
df16 = pd.DataFrame(df16, columns=['ecg','emotion_label'])

check_nan = df16 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

#####################################################################################################################
#s17 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S17\S17.pkl"
with open(file_path, 'rb') as f:
        df_chest = pickle.load(f, encoding="latin1")

chest_data = df_chest['signal']['chest']

ecg = chest_data['ECG']
#print(ecg)

#get the label data
stress_label_chest = df_chest['label']
#print (stress_label_chest)

#make sure both are pandas dataframe
df_ecg = pd.DataFrame(ecg)
df_stress_label_chest = pd.DataFrame(stress_label_chest)

#give columns a name
df_ecg.columns = ['ecg']
df_stress_label_chest.columns = ['emotion_label']
#print(df_ecg)
#print(df_stress_label_chest)

#join columns
df17 = df_ecg.join(df_stress_label_chest)
#print (df)

#remove negative values
df17 = pd.DataFrame(df17)
df17 = df17[df17 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df17)

#remove unneeded data label 0
rows = df17[df17["emotion_label"] == 0].index
df17.drop(rows, inplace=True)
#print(df17)

#remove unneeded data label 5
rows = df17[df17["emotion_label"] == 5].index
df17.drop(rows, inplace=True)
#print(df17)

#remove unneeded data label 6 
rows = df17[df17["emotion_label"] == 6].index
df17.drop(rows, inplace=True)
#print(df17)

#remove unneeded data label 7
rows = df17[df17["emotion_label"] == 7].index
df17.drop(rows, inplace=True)
print("Dataframe after dropping unneeded labels and negative values")
print(df17)

#check for null values
df17 = pd.DataFrame(df17, columns=['ecg','emotion_label'])

check_nan = df17 ['ecg'].isnull().values.any() #result =false
print(check_nan) #no nan

###########################################
##combine

df = pd.concat([df1,df3,df4,df5,df6,df7,df8,df9,df10,df11,df13,df14,df15,df16,df17], ignore_index=True)
print ("Combined table:")
print(df)

# defining x as independent variables, a y as dependent
x = df[['ecg']]
y = df["emotion_label"]

#Train test split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

#traning the model decision tree classifier
#dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
#dt_clf_gini.fit(X_train, y_train)

#evaluate model performance
#y_pred = dt_clf_gini.predict(X_test)
#print (f"Accuracy:", accuracy_score(y_test, y_pred))
#print('Accuracy2: %.3f' % accuracy_score(y_test, y_pred))

#average=weighted means to combine precision values from multiple classes
#precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
#print(f"Precision: {precision}")
#print('Precision2: %.3f' % precision_score(y_test, y_pred, average='weighted', zero_division=0))


#recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
#print(f"Recall: {recall}")
#print('Recall2: %.3f' % recall_score(y_test, y_pred, average='weighted', zero_division=0))

#F1 Score
#print('F1 Score: %.3f' % f1_score(y_test, y_pred, average='weighted', zero_division=0))

#random forest regression
rf_classifier = RandomForestClassifier(n_estimators=35, max_depth=20, min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)



#LightGBM Model

#encoder = LabelEncoder()

# Fit one encoder using all possible labels
#encoder.fit(np.concatenate([
   # np.asarray(y_train),
    #np.asarray(y_test)
#]))

#y_train_encoded = encoder.transform(y_train)
#y_test_encoded = encoder.transform(y_test)

#sns.countplot(x=y_train)
#plt.title("Target Distribution")
#plt.show()

#corr=df.corr()

#plt.figure(figsize=(10,6))
#sns.heatmap(corr,cmap="coolwarm",center=0)
#plt.title("Correlation Matrix")
#plt.show()

#train_data=lgb.Dataset(X_train,label=y_train_encoded)
#valid_data=lgb.Dataset(X_test,label=y_test_encoded)

#params = {
 #   "objective": "multiclass",
  #  "num_class": len(encoder.classes_),
   # "metric": "multi_logloss",
    #"boosting_type": "gbdt",
    #"learning_rate": 0.05,
    #"num_leaves": 31,
  #  "feature_fraction": 0.8,
    #"bagging_fraction": 0.8,
    #"bagging_freq": 5,
    #"min_data_in_leaf": 20,
    #"num_threads": -1,
    #"verbosity": -1
#}


#model=lgb.train(
#params,
#train_data,
#num_boost_round=500,
#valid_sets=[valid_data],
#callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
#)

#y_pred_prob=model.predict(X_test,num_iteration=model.best_iteration)
#y_pred= np.argmax(y_pred_prob, axis=1)

#accuracy=accuracy_score(y_test_encoded,y_pred)
#precision=precision_score(y_test_encoded,y_pred, average='macro', zero_division=0)
#recall=recall_score(y_test_encoded,y_pred, average='macro',zero_division=0)
#f1=f1_score(y_test_encoded,y_pred, average='macro', zero_division=0)
#auc=roc_auc_score(y_test_encoded,y_pred_prob, multi_class="ovr", average="macro")

#print("Accuracy:",accuracy)
#print("Precision:",precision)
#print("Recall:",recall)
#print("F1 Score:",f1)
#print("AUC:",auc)

#print(classification_report(y_test_encoded,y_pred, labels=np.arrange(len(encoder.classes_)), zero_division=0))

#importance=pd.DataFrame({
#"feature":x.columns,
#"importance":model.feature_importance()
#}).sort_values(by="importance",ascending=False)

#sns.barplot(x="importance",y="feature",data=importance.head(10))
#plt.title("Top Features")
#plt.show()

#kf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
#scores=[]

#for train_idx,val_idx in kf.split(x,y_encoded):
 #   X_tr,X_val=x.iloc[train_idx],x.iloc[val_idx]
  #  y_tr,y_val=y_encoded[train_idx], y_encoded[val_idx]

   # train_set=lgb.Dataset(X_tr,label=y_tr)
    #val_set=lgb.Dataset(X_val,label=y_val)

    #cv_model=lgb.train(
     #   params,
      #  train_set,
       # num_boost_round=500,
        #valid_sets=[val_set],
        #callbacks=[
         #   lgb.early_stopping(30, verbose=False),
          #  lgb.log_evaluation(0)
        #]
    #)
#val_pred_prob = cv_model.predict(
 #           X_val,
  #          num_iteration=cv_model.best_iteration
   #  )   
#fold_auc = roc_auc_score(
 #       y_val,
  #      val_pred_prob,
   #     multi_class="ovr",
    #    average="macro"
    #)

#scores.append(fold_auc)

#print("CV AUC:", np.mean(scores))
#save the model
#with open ("WESADECG.pkl", "wb") as model_file:
 #   pickle.dump(rf_classifier, model_file)