import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#s2 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S2\S2.pkl"


with open(file_path, 'rb') as f:
        df1 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df1['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df1['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df1 = df_bvp.join(df_stress_label)
#print (df1)

#remove negative values
df1 = pd.DataFrame(df1)
df1 = df1[df1 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df1)

#remove unneeded data label 0
rows = df1[df1["emotion_label"] == 0].index
df1.drop(rows, inplace=True)
#print(df1)

#remove unneeded data label 5
rows = df1[df1["emotion_label"] == 5].index
df1.drop(rows, inplace=True)
#print(df1)

#remove unneeded data label 6
rows = df1[df1["emotion_label"] == 6].index
df1.drop(rows, inplace=True)
#print(df1)

#remove unneeded data label 7
rows = df1[df1["emotion_label"] == 7].index
df1.drop(rows, inplace=True)
print ("S2 Dataframe after removing negative values and irrelevant labels:")
print(df1)

#############################################################################################

#s3 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S3\S3.pkl"


with open(file_path, 'rb') as f:
        df2 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df2['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df2['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df2 = df_bvp.join(df_stress_label)
#print (df2)

#remove negative values
df2 = pd.DataFrame(df2)
df2 = df2[df2 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df2)

#remove unneeded data label 0
rows = df2[df2["emotion_label"] == 0].index
df2.drop(rows, inplace=True)
#print(df2)

#remove unneeded data label 5
rows = df2[df2["emotion_label"] == 5].index
df2.drop(rows, inplace=True)
#print(df2)

#remove unneeded data label 6
rows = df2[df2["emotion_label"] == 6].index
df2.drop(rows, inplace=True)
#print(df2)

#remove unneeded data label 7
rows = df1[df1["emotion_label"] == 7].index
df2.drop(rows, inplace=True)
print ("S3 Dataframe after removing negative values and irrelevant labels:")
print(df2)

########################################################################################################

#s4 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S4\S4.pkl"


with open(file_path, 'rb') as f:
        df3 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df3['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df3['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df3 = df_bvp.join(df_stress_label)
#print (df3)

#remove negative values
df3 = pd.DataFrame(df3)
df3 = df3[df3 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df3)

#remove unneeded data label 0
rows = df3[df3["emotion_label"] == 0].index
df3.drop(rows, inplace=True)
#print(df3)

#remove unneeded data label 5
rows = df3[df3["emotion_label"] == 5].index
df3.drop(rows, inplace=True)
#print(df3)

#remove unneeded data label 6
rows = df3[df3["emotion_label"] == 6].index
df3.drop(rows, inplace=True)
#print(df3)

#remove unneeded data label 7
rows = df1[df1["emotion_label"] == 7].index
df3.drop(rows, inplace=True)
print ("S4 Dataframe after removing negative values and irrelevant labels:")
print(df3)

#############################################################################################

#s5 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S5\S5.pkl"


with open(file_path, 'rb') as f:
        df15 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df15['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df15['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df15 = df_bvp.join(df_stress_label)
#print (df15)

#remove negative values
df15 = pd.DataFrame(df15)
df15 = df15[df15 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df15)

#remove unneeded data label 0
rows = df15[df15["emotion_label"] == 0].index
df15.drop(rows, inplace=True)
#print(df15)

#remove unneeded data label 5
rows = df15[df15["emotion_label"] == 5].index
df15.drop(rows, inplace=True)
#print(df15)

#remove unneeded data label 6
rows = df15[df15["emotion_label"] == 6].index
df15.drop(rows, inplace=True)
#print(df15)

#remove unneeded data label 7
rows = df15[df15["emotion_label"] == 7].index
df15.drop(rows, inplace=True)
print ("S5 Dataframe after removing negative values and irrelevant labels:")
print(df15)

########################################################################################################

#s6 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S6\S6.pkl"


with open(file_path, 'rb') as f:
        df4 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df4['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df4['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df4 = df_bvp.join(df_stress_label)
#print (df4)

#remove negative values
df4 = pd.DataFrame(df4)
df4 = df4[df4 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df4)

#remove unneeded data label 0
rows = df4[df4["emotion_label"] == 0].index
df4.drop(rows, inplace=True)
#print(df4)

#remove unneeded data label 5
rows = df4[df4["emotion_label"] == 5].index
df4.drop(rows, inplace=True)
#print(df4)

#remove unneeded data label 6
rows = df4[df4["emotion_label"] == 6].index
df4.drop(rows, inplace=True)
#print(df4)

#remove unneeded data label 7
rows = df4[df4["emotion_label"] == 7].index
df4.drop(rows, inplace=True)
print ("S6 Dataframe after removing negative values and irrelevant labels:")
print(df4)

########################################################################################################

#s7 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S7\S7.pkl"


with open(file_path, 'rb') as f:
        df5 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df5['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df5['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df5 = df_bvp.join(df_stress_label)
#print (df5)

#remove negative values
df5 = pd.DataFrame(df5)
df5 = df5[df5 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df4)

#remove unneeded data label 0
rows = df5[df5["emotion_label"] == 0].index
df5.drop(rows, inplace=True)
#print(df5)

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
print ("S7 Dataframe after removing negative values and irrelevant labels:")
print(df5)

########################################################################################################

#s8 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S8\S8.pkl"


with open(file_path, 'rb') as f:
        df6 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df6['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df6['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df6 = df_bvp.join(df_stress_label)
#print (df5)

#remove negative values
df6 = pd.DataFrame(df6)
df6 = df6[df6 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df6)

#remove unneeded data label 0
rows = df6[df6["emotion_label"] == 0].index
df6.drop(rows, inplace=True)
#print(df5)

#remove unneeded data label 5
rows = df6[df6["emotion_label"] == 5].index
df6.drop(rows, inplace=True)
#print(df5)

#remove unneeded data label 6
rows = df6[df6["emotion_label"] == 6].index
df6.drop(rows, inplace=True)
#print(df5)

#remove unneeded data label 7
rows = df6[df6["emotion_label"] == 7].index
df6.drop(rows, inplace=True)
print ("S8 Dataframe after removing negative values and irrelevant labels:")
print(df6)

########################################################################################################

#s9 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S9\S9.pkl"


with open(file_path, 'rb') as f:
        df7 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df7['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df7['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df7 = df_bvp.join(df_stress_label)
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
#print(df5)

#remove unneeded data label 6
rows = df7[df7["emotion_label"] == 6].index
df7.drop(rows, inplace=True)
#print(df7)

#remove unneeded data label 7
rows = df7[df7["emotion_label"] == 7].index
df7.drop(rows, inplace=True)
print ("S9 Dataframe after removing negative values and irrelevant labels:")
print(df7)

########################################################################################################

#s10 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S10\S10.pkl"


with open(file_path, 'rb') as f:
        df8 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df8['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df8['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df8 = df_bvp.join(df_stress_label)
#print (df8)

#remove negative values
df8 = pd.DataFrame(df8)
df8 = df8[df8 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df8)

#remove unneeded data label 0
rows = df8[df8["emotion_label"] == 0].index
df8.drop(rows, inplace=True)
#print(df8)

#remove unneeded data label 5
rows = df8[df8["emotion_label"] == 5].index
df8.drop(rows, inplace=True)
#print(df8)

#remove unneeded data label 6
rows = df8[df8["emotion_label"] == 6].index
df8.drop(rows, inplace=True)
#print(df8)

#remove unneeded data label 7
rows = df8[df8["emotion_label"] == 7].index
df8.drop(rows, inplace=True)
print ("S10 Dataframe after removing negative values and irrelevant labels:")
print(df8)

########################################################################################################

#s11 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S11\S11.pkl"


with open(file_path, 'rb') as f:
        df9 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df9['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df9['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df9 = df_bvp.join(df_stress_label)
#print (df9)

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
print ("S11 Dataframe after removing negative values and irrelevant labels:")
print(df9)

########################################################################################################

#s13 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S13\S13.pkl"


with open(file_path, 'rb') as f:
        df10 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df10['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df10['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df10 = df_bvp.join(df_stress_label)
#print (df10)

#remove negative values
df10 = pd.DataFrame(df10)
df10 = df10[df10 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df10)

#remove unneeded data label 0
rows = df10[df10["emotion_label"] == 0].index
df10.drop(rows, inplace=True)
#print(df10)

#remove unneeded data label 5
rows = df10[df10["emotion_label"] == 5].index
df10.drop(rows, inplace=True)
#print(df10)

#remove unneeded data label 6
rows = df10[df10["emotion_label"] == 6].index
df10.drop(rows, inplace=True)
#print(df10)

#remove unneeded data label 7
rows = df10[df10["emotion_label"] == 7].index
df10.drop(rows, inplace=True)
print ("S13 Dataframe after removing negative values and irrelevant labels:")
print(df10)

########################################################################################################

#s14 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S14\S14.pkl"


with open(file_path, 'rb') as f:
        df11 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df11['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df11['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df11 = df_bvp.join(df_stress_label)
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
print ("S14 Dataframe after removing negative values and irrelevant labels:")
print(df11)

########################################################################################################

#s15 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S15\S15.pkl"


with open(file_path, 'rb') as f:
        df12 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df12['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df12['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df12 = df_bvp.join(df_stress_label)
#print (df12)

#remove negative values
df12 = pd.DataFrame(df12)
df12 = df12[df12 >=0].dropna()
#print("DataFrame after dropping negative values:")
#print(df12)

#remove unneeded data label 0
rows = df12[df12["emotion_label"] == 0].index
df12.drop(rows, inplace=True)
#print(df12)

#remove unneeded data label 5
rows = df12[df12["emotion_label"] == 5].index
df12.drop(rows, inplace=True)
#print(df12)

#remove unneeded data label 6
rows = df12[df12["emotion_label"] == 6].index
df12.drop(rows, inplace=True)
#print(df12)

#remove unneeded data label 7
rows = df12[df12["emotion_label"] == 7].index
df12.drop(rows, inplace=True)
print ("S15 Dataframe after removing negative values and irrelevant labels:")
print(df12)

########################################################################################################

#s16 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S16\S16.pkl"


with open(file_path, 'rb') as f:
        df13 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df13['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df13['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df13 = df_bvp.join(df_stress_label)
#print (df13)

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
print ("S16 Dataframe after removing negative values and irrelevant labels:")
print(df13)

########################################################################################################

#s17 pkl file
file_path = r"C:\Users\m45m4\Documents\Computing project\WESAD\WESAD\S17\S17.pkl"


with open(file_path, 'rb') as f:
        df14 = pickle.load(f, encoding="latin1")

#print (df) 

#get the wrist data
wrist_data = df14['signal']['wrist']
bvp = wrist_data['BVP']

#get the label data
emotion_label = df14['label']

#make sure both are pandas dataframe
df_bvp = pd.DataFrame(bvp)
df_stress_label = pd.DataFrame(emotion_label)

#give columns a name
df_bvp.columns = ['BVP']
df_stress_label.columns = ['emotion_label']


#join columns
df14 = df_bvp.join(df_stress_label)
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
#print(df14)

#remove unneeded data label 6
rows = df14[df14["emotion_label"] == 6].index
df14.drop(rows, inplace=True)
#print(df14)

#remove unneeded data label 7
rows = df14[df14["emotion_label"] == 7].index
df14.drop(rows, inplace=True)
print ("S17 Dataframe after removing negative values and irrelevant labels:")
print(df14)

df = pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15], ignore_index=True)
print ("Combined table:")
print(df)

# defining x as independent variables, a y as dependent
x = df[['BVP']]
y = df['emotion_label']

#Train test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#traning the model
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)

#evaluate model performance
y_pred = dt_clf_gini.predict(X_test)
print (f"Accuracy:", accuracy_score(y_test, y_pred))
print('Accuracy2: %.3f' % accuracy_score(y_test, y_pred))

precision = precision_score(y_test, y_pred)
print(f"Precision: {precision}")
print('Precision2: %.3f' % precision_score(y_test, y_pred))

recall = recall_score(y_test, y_pred)
print(f"Recall: {recall}")
print('Recall2: %.3f' % recall_score(y_test, y_pred))

#F1 Score
print('F1 Score: %.3f' % f1_score(y_test, y_pred))

#save the model
with open ("WESADBVP.pkl", "wb") as model_file:
    pickle.dump(dt_clf_gini, model_file)