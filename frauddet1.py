import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

df= pd.read_csv('creditcard.csv')
#print(df.head())
#print('Columns:\n',df.columns,'\n')
#print(df.describe())
print('\nThere are 28 transformed columns and 2 original columns: Time and Amount as the features\n')
print('There are ',df.isnull().any().sum(), ' null values.\n')

Frauds= df[df['Class']== 1]
Not_frauds= df[df['Class']== 0]
print('\nNo. of Fraudulent Transactions= ', len(Frauds)/len(df)*100, '%')
print('\nOne could get an accuracy of ', 1-len(Frauds)/len(df)*100, '%', ' by predicting all are fraudulent transactions.')
print('\nSo recall rate (True positives/(True Positives+False Negatives)) would be the metric to optimize.')

sns.countplot(df.Class)
plt.show()

plt.figure(figsize= (20, 20))
sns.heatmap(df.corr(), cmap= 'Blues')
plt.show()
print('\nThe features V1-V28 are not correlated as they are obtained and not originally given with the data\n')

plt.xlim(0, 3000)
sns.kdeplot(df.Amount)
plt.show()


df['Scaled_Amount']= StandardScaler().fit_transform(df['Amount'].reshape(-1, 1))
#print(df[['Amount', 'Scaled_Amount']].head())
Frauds['Scaled_Amount']= StandardScaler().fit_transform(Frauds['Amount'].reshape(-1, 1))
Not_frauds['Scaled_Amount']= StandardScaler().fit_transform(Not_frauds['Amount'].reshape(-1, 1))
sns.kdeplot(Frauds['Scaled_Amount'],shade=True,color="red")
plt.show()
sns.kdeplot(Not_frauds['Scaled_Amount'],shade=True,color="green")
plt.show()

print('\nDetermining when do people shop:\n')
#print(df[['Time']])
df['Time_in_hours']= df['Time']/3600
#print(df['Time_in_hours'])
plt.figure(figsize= (10, 10))
plt.ylim(0, 0.040)
sns.kdeplot(df['Time_in_hours'])
plt.show()
#plt.xticks([0, 1])
'''
sns.violinplot(data= df, y= 'Time_in_hours', x= 'Class')
plt.show()
sns.boxplot(data= df, y= 'Time_in_hours', x= 'Class')
plt.show()
'''
#sns.jointplot(df['Time-in_hours'], df['Class'])
df_byTime = df.groupby(['Class', 'Time_in_hours'])['Amount'].count()
#print('Valid Transactions grouped by time\n:',df_byTime[0].head())
#print('Invalid Transactions grouped by time\n:',df_byTime[1].head())
#df_byTime[0].plot.bar(title = 'Valid transactions by Hour', legend = True)
#plt.show()
#df_byTime[1].plot.bar(title= 'Invalid transactions by Hour', legend= True)
#plt.show()
print('Now we find out that whether the amount spent on the fraudulent transactions\
      is significantly larger than the Normal Transactions: (Going with 99% significance level)\n');
x= df[df.Class == 0].Amount
y= df[df.Class == 1].Amount
y_size= len(y)
y_mean= y.mean()
x_std= x.std()
x_mean= x.mean()
z_score= (y_mean- x_mean)/(x_std/y_size**0.5)
print('The Z score of the above formulation is: ', z_score, '\n')
print('As the z-score is more than 2.326 we reject this hypothesis.\n')

print('Doing a two-tailed test so Level of Signifiacance is 3.37\n')
'''
Cols_generated= [i for i in df.columns if 'V' in i]
fraud_size= len(Frauds)
for i in Cols_generated:
    mean= Not_frauds[i].mean()
    std= Not_frauds[i].std()
    z_score= (Frauds[i].mean()-mean)/(std/fraud_size**0.5)
    print(i, 'is significant' if abs(z_score)>=3.37 else 'insignificant')
print('So, we find out that columns V13, V15, V22, V23, V25, V26 are insignificant.\n')
print('\nSo maybe removing them will lead to a better result.')
'''
doubt_cols1= ['Amount', 'Scaled_Amount']
doubt_cols2= ['V13', 'V15', 'V22', 'V23', 'V25', 'V26']
#Preferred: Scaled Amount
#plt.scatter(x= 'Class', y= doubt_cols1[1], data= df)
#plt.show()
#plt.scatter(x= 'Class', y= doubt_cols1[1], data= df)
#plt.show()
df.drop(['Time', 'Time_in_hours', 'Amount'], axis= 1, inplace= True)
#print(df.columns)
'''
for i in doubt_cols2:
    print('We are talking about ', i, '\n')
    plt.figure(figsize= (4,4))
    plt.xticks([0, 1])
    sns.violinplot(x= 'Class', y= i, data= df)
    plt.show()
    print('\n')
'''
'''
#Insignificant: V22
df.drop(['V22'], axis= 1, inplace= True)
X= np.array(df.drop(['Class'], axis= 1))
y= np.array(df['Class'])

x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state= 14)

rf= RandomForestClassifier(random_state= 14, n_estimators= 100)
rf.fit(x_train, y_train)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
print(classification_report(y_test,rf.predict(x_test)))
'''
'''
Dropping V22:
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.93      0.77      0.84       115

avg / total       1.00      1.00      1.00     71202
'''
'''
Using V22:
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.91      0.76      0.82       115

avg / total       1.00      1.00      1.00     71202
'''
'''
sv=SVC(kernel='rbf')
sv.fit(x_train,y_train)
print(classification_report(y_test, sv.predict(x_test)))
'''
'''
SVC:
kernel= 'rbf'
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.92      0.69      0.79       115

avg / total       1.00      1.00      1.00     71202
'''
'''
RF with 1000 trees:
            precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.91      0.82      0.86       115

avg / total       1.00      1.00      1.00     71202

pickle.dump(rf, open(filename, 'wb'))
'''
df.drop(['V22', 'V23', 'V25', 'V13'], axis= 1, inplace= True)
#rf= RandomForestClassifier(random_state= 14, n_estimators= 100)

X= np.array(df.drop(['Class'], axis= 1))
y= np.array(df.Class)
x_train, x_test,y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state= 14)
#rf.fit(x_train, y_train)
filename = 'finalized_model_2.sav'
#pickle.dump(rf, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
y_pred= loaded_model.predict(x_test)
#print(classification_report(y_test, y_pred))
'''
            precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.91      0.82      0.86       115

avg / total       1.00      1.00      1.00     71202
'''
'''
cm= pd.crosstab(y_test, y_pred, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm, annot=True, fmt= '')
plt.show()
'''
Frauds_upsampled= resample(Frauds, replace= True, n_samples= 284315, random_state= 14)
df_upsampled= pd.concat([Not_frauds, Frauds_upsampled])
#print(df_upsampled.Class.value_counts())
X_upsampled= df_upsampled.drop(['Class'], axis =1)
y_upsampled= df_upsampled.Class
x_train_u, x_test_u, y_train_u, y_test_u= train_test_split(X_upsampled, y_upsampled, test_size= 0.25, random_state= 14)
'''
rf_u= RandomForestClassifier(random_state =14, n_estimators= 100)
rf_u.fit(x_train_u, y_train_u)
filename_1= 'finalized_model_1.sav'
pickle.dump(rf_u, open(filename_1, 'wb'))
'''
filename_1= 'finalized_model_1.sav'
loaded_model_1= pickle.load(open(filename_1, 'rb'))
y_pred_u= loaded_model_1.predict(x_test_u)
#print(classification_report(y_test_u, y_pred_u))
'''
1    284315
0    284315
Name: Class, dtype: int64
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     70956
          1       1.00      1.00      1.00     71202

avg / total       1.00      1.00      1.00    142158

[70950 71208]
'''
#print(np.bincount(y_pred_u))
Not_frauds_downsampled= resample(Not_frauds,replace=False,n_samples=492,random_state=14)
df_downsampled = pd.concat([Not_frauds_downsampled, Frauds])
#print(df_downsampled.Class.value_counts())
X_d= df_downsampled.drop(['Class'], axis =1)
y_d= df_downsampled.Class
x_train_d, x_test_d, y_train_d, y_test_d= train_test_split(X_d, y_d, test_size= 0.25, random_state= 14)
'''
rf_d= RandomForestClassifier(random_state =14, n_estimators= 100)
rf_d.fit(x_train_d, y_train_d)
filename_2= 'finalized_model.sav'
pickle.dump(rf_d, open(filename_2, 'wb'))
'''
'''
filename_2= 'finalized_model.sav'
loaded_model_2= pickle.load(open(filename_2, 'rb'))
y_pred_d= loaded_model_2.predict(x_test_d)
#print(classification_report(y_test_d, y_pred_d))
#print(np.bincount(y_pred_d))
'''
'''
1    492
0    492
Name: Class, dtype: int64
             precision    recall  f1-score   support

          0       0.96      1.00      0.98       120
          1       1.00      0.96      0.98       126

avg / total       0.98      0.98      0.98       246

[125 121]
'''
'''
cm_= pd.crosstab(y_test_u, y_pred_u, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm_, annot=True, fmt= '')
plt.show()
cm__= pd.crosstab(y_test_d, y_pred_d, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm__, annot=True, fmt= '')
plt.show()
'''
'''
#Plotting the ROC-AUC Curve of the sampled dataframe
print('Plotting the ROC-AUC Curve of the sampled dataframe')
y_pred_prob= loaded_model.predict_proba(x_test)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#AUC= 0.95
#Plotting the ROC-AUC Curve of the oversampled dataframe
print('Plotting the ROC-AUC Curve of the oversampled dataframe')
y_pred_prob_u= loaded_model_1.predict_proba(x_test_u)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test_u,y_pred_prob_u)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#AUC=1.00
#Plotting the ROC-AUC Curve of the undersampled dataframe
print('Plotting the ROC-AUC Curve of the undersampled dataframe')
y_pred_prob_d= loaded_model_2.predict_proba(x_test_d)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test_d,y_pred_prob_d)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#AUC= 0.99
'''









'''
print(classification_report(y_test,loaded_model.predict(x_test)))
importances = loaded_model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize= (20, 20))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), df.drop(['Class'], axis= 1).columns[indices])
plt.xlabel('Relative Importance')
plt.show()

'''
'''
filename_1= 'finalized_model_1.sav'
'''
'''
1    284315
0    284315
Name: Class, dtype: int64
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     70956
          1       1.00      1.00      1.00     71202

avg / total       1.00      1.00      1.00    142158
Y_pred:
[70950 71208]
'''
'''
'''
'''
loaded_model = pickle.load(open(filename_1, 'rb'))
y_pred_d= loaded_model.predict(x_test_d)
print(classification_report(y_test_d, y_pred_d))
print(np.bincount(y_pred_d))
'''
'''
1    492
0    492
Name: Class, dtype: int64
             precision    recall  f1-score   support

          0       0.96      1.00      0.98       120
          1       1.00      0.96      0.98       126

avg / total       0.98      0.98      0.98       246

[125 121]
'''
'''
filename_1= 'finalized_model.sav'
loaded_model = pickle.load(open(filename_1, 'rb'))
y_pred= loaded_model.predict(x_test)
cm= pd.crosstab(y_test, y_pred, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm, annot=True, fmt= '')
plt.show()

filename_2= 'finalized_model_1.sav'
loaded_model_1= pickle.load(open(filename_2, 'rb'))
y_pred_1= loaded_model_1.predict(x_test_)
'''
'''
print('Using Decision Tree Classifier:\n')
dt= DecisionTreeClassifier(random_state= 14)
#, class_weight= 'balanced')
dt.fit(x_train, y_train)
y_pred= dt.predict(x_test)
print(classification_report(y_test, y_pred))
'''
'''
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.72      0.76      0.74       115

avg / total       1.00      1.00      1.00     71202
'''
'''
cm= pd.crosstab(y_test, y_pred, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm, annot=True, fmt= '')
plt.show()
#Plotting the ROC-AUC Curve of the sampled dataframe
print('Plotting the ROC-AUC Curve of the sampled dataframe')
y_pred_prob= dt.predict_proba(x_test)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''
'''
dt.fit(x_train_u, y_train_u)
y_pred_u= dt.predict(x_test_u)
print(classification_report(y_test_u, y_pred_u))
'''
'''
            precision    recall  f1-score   support

          0       1.00      1.00      1.00     70956
          1       1.00      1.00      1.00     71202

avg / total       1.00      1.00      1.00    142158
'''
'''
cm1= pd.crosstab(y_test_u, y_pred_u, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm1, annot=True, fmt= '')
plt.show()
#Plotting the ROC-AUC Curve of the oversampled dataframe
print('Plotting the ROC-AUC Curve of the sampled dataframe')
y_pred_prob_u= dt.predict_proba(x_test_u)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test_u,y_pred_prob_u)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

dt.fit(x_train_d, y_train_d)
y_pred_d= dt.predict(x_test_d)
print(classification_report(y_test_d, y_pred_d))
'''
'''
             precision    recall  f1-score   support

          0       0.97      0.97      0.97       120
          1       0.98      0.98      0.98       126

avg / total       0.98      0.98      0.98       246
'''
'''
cm2= pd.crosstab(y_test_d, y_pred_d, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm2, annot=True, fmt= '')
plt.show()
#Plotting the ROC-AUC Curve of the undersampled dataframe
print('Plotting the ROC-AUC Curve of the undersampled dataframe')
y_pred_prob_d= dt.predict_proba(x_test_d)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test_d,y_pred_prob_d)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''
'''
#Checking out the class_weight= 'balanced' parameter:
rf_= RandomForestClassifier(class_weight= 'balanced', n_estimators= 100, random_state= 14)
rf_.fit(x_train, y_train)
y_pred_= rf_.predict(x_test)
print(classification_report(y_test,y_pred_))
cm_11= pd.crosstab(y_test, y_pred_, rownames= ['True'], colnames= ['False'])
sns.heatmap(cm_11, annot=True, fmt= '')
plt.show()
print('Plotting the ROC-AUC Curve for the class_weights= balanced dataframe')
y_pred_prob_= rf_.predict_proba(x_test)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob_)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''
'''
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.94      0.81      0.87       115

avg / total       1.00      1.00      1.00     71202
'''
'''
#AUC= 0.96
print('Using SMOTE')
x_train_s, x_val_s, y_train_s, y_val_s= train_test_split(x_train, y_train, test_size = 0.1, random_state=14)
sm=SMOTE(random_state= 14, ratio= 1.0)
x_train_res, y_train_res= sm.fit_sample(x_train_s, y_train_s)
rf_smote= RandomForestClassifier(random_state= 14, n_estimators= 100)
rf_smote.fit(x_train_res, y_train_res)
'''
'''
print('Validation Results:')
print(classification_report(y_val_s, rf_smote.predict(x_val_s)))
print('Test results')
print(classification_report(y_test, rf_smote.predict(x_test)))
Validation Results:
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     21325
          1       0.88      0.81      0.84        36

avg / total       1.00      1.00      1.00     21361

Test results
             precision    recall  f1-score   support

          0       1.00      1.00      1.00     71087
          1       0.83      0.83      0.83       115

avg / total       1.00      1.00      1.00     71202
'''
'''
cm_111= pd.crosstab(y_test, rf_smote.predict(x_test), rownames= ['True'], colnames= ['False'])
sns.heatmap(cm_111, annot=True, fmt= '')
plt.show()
print('Plotting the ROC-AUC Curve for the class_weights= balanced dataframe')
y_pred_prob_= rf_smote.predict_proba(x_test)[:, 1].ravel()
fpr, tpr, thresholds = roc_curve(y_test,y_pred_prob_)
roc_auc = auc(fpr,tpr)
# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#AUC= 0.98
'''
'''
Code doesn't run:
    Don't know why!!
us= NearMiss(ratio= 1, version=1, random_state= 14)
x_train_nm1, y_train_nm1= us.fit_sample(x_train, y_train)
rf= RandomForestClassifier(n_estimators= 100, random_state= 14)
rf.fit(x_train_nm1, y_train_nm1)
y_pred_nm1= rf.predict(x_test)
print(classification_report(y_test,y_pred_nm1))

us= NearMiss(ratio= 1, version=2, random_state= 14)
x_train_nm1, y_train_nm1= us.fit_sample(x_train, y_train)
rf= RandomForestClassifier(n_estimators= 100, random_state= 14)
rf.fit(x_train_nm1, y_train_nm1)
y_pred_nm1= rf.predict(x_test)
print(classification_report(y_test,y_pred_nm1))

us= NearMiss(ratio= 1, version=3, random_state= 14)
x_train_nm1, y_train_nm1= us.fit_sample(x_train, y_train)
rf= RandomForestClassifier(n_estimators= 100, random_state= 14)
rf.fit(x_train_nm1, y_train_nm1)
y_pred_nm1= rf.predict(x_test)
print(classification_report(y_test,y_pred_nm1))
'''

'''
from sklearn.preprocessing import StandardScaler

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time','Amount'],axis=1)
data.head()

X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']

number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))



from sklearn.cross_validation import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample
                                                                                                   ,y_undersample
                                                                                                   ,test_size = 0.3
                                                                                                   ,random_state = 0)
print("")
print("Number transactions train dataset: ", len(X_train_undersample))
print("Number transactions test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))




lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test_undersample.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.figure(figsize=(10,10))

j = 1
for i in thresholds:
    y_test_predictions_high_recall = y_pred_undersample_proba[:,1] > i
    
    plt.subplot(3,3,j)
    j += 1
    
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test_undersample,y_test_predictions_high_recall)
    np.set_printoptions(precision=2)

    print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

    # Plot non-normalized confusion matrix
    class_names = [0,1]
    plot_confusion_matrix(cnf_matrix
                          , classes=class_names
                          , title='Threshold >= %s'%i) 
    
    
    



'''




'''
class0train = class0.iloc[0:6000]
class1train = class1

# combine subset of different classes into one balaced dataframe
train = class0train.append(class1train, ignore_index=True).values


X = train[:,0:30].astype(float)
Y = train[:,30]


model = XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)

# use area under the precision-recall curve to show classification accuracy
scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring = scoring)
print( "AUC: %.3f (%.3f)" % (results.mean(), results.std()) )



fig_size = plt.rcParams["figure.figsize"] # Get current size

old_fig_params = fig_size
# new figure parameters
fig_size[0] = 12
fig_size[1] = 9
   
plt.rcParams["figure.figsize"] = fig_size # set new size


# plot roc-curve
# code adapted from http://scikit-learn.org
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)

colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
lw = 2

i = 0
for (train, test), color in zip(kfold.split(X, Y), colors):
    probas_ = model.fit(X[train], Y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(Y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= kfold.get_n_splits(X, Y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()





'''