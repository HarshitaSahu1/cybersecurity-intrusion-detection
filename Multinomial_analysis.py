import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math


finalized_table_multi = pd.read_csv('finalized_table.csv')
finalized_table_multi


## No missings
print(finalized_table_multi.isna().sum())


## No Duplications 
print(finalized_table_multi[finalized_table_multi.duplicated()]) 

finalized_table_multi

### Outliers clip 

for col in finalized_table_multi.select_dtypes(['int64','float64']).columns:
    finalized_table_multi[col] =  finalized_table_multi[col].clip(upper = finalized_table_multi[col].quantile(0.999),lower = finalized_table_multi[col].quantile(0.001))


### Distribution of Target 

attack_dist = finalized_table_multi.groupby('Attack').size().reset_index(name = 'attack_counts')

fig,axes = plt.subplots(figsize = (10,5))
axes.bar(x= attack_dist['Attack'],height = attack_dist['attack_counts'])
axes.get_xlabel()
axes.tick_params(axis = 'x',rotation = 45)
axes.title('Distribution Of Attacks')
axes.show()


### Correlation with Target

corr_data = finalized_table_multi.copy()

from sklearn.preprocessing import LabelEncoder
labels = LabelEncoder()
corr_data['Attack'] = labels.fit_transform(corr_data['Attack'])

corr_matrix = corr_data[corr_data.select_dtypes(['int64','float64']).columns].corr()

sns.heatmap(corr_matrix[['Attack']],annot= True)
plt.show()

### duration 
cols = 2
rows = int(len(finalized_table_multi['Attack'].unique())/2)
fig,axes = plt.subplots(rows,cols,figsize = (20,5))
axes = axes.flatten()
count = 0
for i in finalized_table_multi['Attack'].unique():
    temp = finalized_table_multi[finalized_table_multi['Attack'] == i]
    if temp['duration'].nunique()>1:
        sns.histplot(x = temp['duration'],ax = axes[count],kde = True,bins = 10,hue = temp['Attack'])
        
    else:
        sns.histplot(x = temp['duration'],ax = axes[count],kde = False,bins = 10,hue = temp['Attack'])

    axes[count].set_title(f'{i} Distribution Based On Durations')
    count +=1
plt.tight_layout()
plt.show()


X = finalized_table_multi.drop(columns = 'Attack')
Y = finalized_table_multi['Attack']

finalized_table_multi.columns = finalized_table_multi.columns.str.strip()

finalized_table_multi['Attack'].value_counts()

### as data is highly imbalanced
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

X_train,X_valid, Y_train,Y_valid = train_test_split(X,Y,test_size = 0.2,random_state= 101,stratify= Y)

smote_ = SMOTE(random_state= 102)

X_train_resampled, Y_train_resampled = smote_.fit_resample(X_train,Y_train)

print('resample_started !')
X_train_resampled.shape
Y_train_resampled.shape
print('resample_finished !')

### before removing columns the accuracy and all 

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score,confusion_matrix
random_before = RandomForestClassifier(n_estimators= 50,random_state= 101)
#rfe = RFE(random_before,n_features_to_select= 10)
rfe = random_before.fit(X_train_resampled,Y_train_resampled)

print(X_train_resampled.shape)
print(Y_train_resampled.shape)
random_before_predict = rfe.predict(X_train_resampled)
print(random_before_predict.shape)


### checking accuracy on trained data
print(accuracy_score(Y_train_resampled,random_before_predict))
print(confusion_matrix(Y_train_resampled,random_before_predict))

### checking on testing data 
print(accuracy_score(Y_valid,random_before.predict(X_valid)))
print(confusion_matrix(Y_valid,random_before.predict(X_valid)))
print(classification_report(Y_valid,random_before.predict(X_valid)))

### visuals of confusion matrixs 
sns.heatmap(confusion_matrix(Y_valid,random_before.predict(X_valid)),annot= True,fmt = 'd',cmap = 'Blues',xticklabels= ['Attack','Normal'],yticklabels=['Attack','Normal'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion_Matrix')
plt.show()

print(random_before_predict.shape)

type(Y_train_resampled)
