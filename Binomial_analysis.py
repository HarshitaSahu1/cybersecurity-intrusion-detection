import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

bufferoverflow = pd.read_csv('Data_of_Attack_Back_BufferOverflow.csv')
ftpwrite = pd.read_csv('Data_of_Attack_Back_FTPWrite.csv',header = None)
guesspassword = pd.read_csv('Data_of_Attack_Back_GuessPassword.csv')
neptune = pd.read_csv('Data_of_Attack_Back_Neptune.csv')
nmap = pd.read_csv('Data_of_Attack_Back_NMap.csv')
normal = pd.read_csv('Data_of_Attack_Back_Normal.csv')
portsweep = pd.read_csv('Data_of_Attack_Back_PortSweep.csv')
rootkit = pd.read_csv('Data_of_Attack_Back_RootKit.csv')
satan = pd.read_csv('Data_of_Attack_Back_Satan.csv')
smurf = pd.read_csv('Data_of_Attack_Back_Smurf.csv')
back = pd.read_csv('Data_of_Attack_Back.csv')

ftpwrite.columns = back.columns


bufferoverflow['Attack'] = 'bufferoverflow'
ftpwrite['Attack'] = 'ftpwrite'
guesspassword['Attack'] = 'guesspassword'
neptune['Attack'] = 'neptune'
nmap['Attack'] = 'nmap'
normal['Attack'] = 'normal'
portsweep['Attack'] = 'portsweep'
rootkit['Attack'] = 'rootkit'
satan['Attack'] = 'satan'
smurf['Attack'] = 'smurf'
back['Attack'] = 'back'



finalized_table = pd.concat([bufferoverflow,ftpwrite,guesspassword,neptune,normal,portsweep,rootkit,satan,smurf,back],axis = 0,ignore_index= True)
print(finalized_table.shape)

finalized_table.to_csv('finalized_table.csv')


## No missings
print(finalized_table.isna().sum())


## No Duplications 
print(finalized_table[finalized_table.duplicated()]) 

## for binomial 

finalized_table['Binomial'] = np.where(finalized_table['Attack'] == 'normal','Normal','Attack')

finalized_table.columns = finalized_table.columns.str.strip()

Binomial_datasets = finalized_table[finalized_table.columns[~finalized_table.columns.isin(['Attack'])]]

### Outliers clip 

for col in Binomial_datasets.select_dtypes(['int64','float64']).columns:
    Binomial_datasets[col] =  Binomial_datasets[col].clip(upper = Binomial_datasets[col].quantile(0.999),lower = Binomial_datasets[col].quantile(0.001))


### Distribution of Target 

attack_dist = Binomial_datasets.groupby('Binomial').size().reset_index(name = 'attack_counts')

plt.pie(attack_dist['attack_counts'],labels= attack_dist['Binomial'],autopct= '%1.1f%%')
plt.title('Distribution Of Attacks')
plt.show()

### Correlation with Target

corr_data = Binomial_datasets.copy()

from sklearn.preprocessing import LabelEncoder
labels = LabelEncoder()
corr_data['Binomial'] = labels.fit_transform(corr_data['Binomial'])

corr_matrix = corr_data[corr_data.select_dtypes(['int64','float64']).columns].corr()

sns.heatmap(corr_matrix[['Binomial']],annot= True)
plt.show()

### duration 

fig,axes = plt.subplots(2,1,figsize = (10,2))
normal = Binomial_datasets[Binomial_datasets['Binomial'] == 'Normal']
attack = Binomial_datasets[Binomial_datasets['Binomial'] == 'Attack']

sns.histplot(x = normal['duration'],ax = axes[0],kde = True,bins = 10,hue = normal['Binomial'])
axes[0].set_title('Normal Distribution Based On Duration')
sns.histplot(x = attack['duration'],ax = axes[1],kde = True,bins = 10,hue = attack['Binomial'])
axes[1].set_title('Attack Distribution Based On Duration')
plt.show()

### count

fig,axes = plt.subplots(1,2,figsize = (10,2))
normal = Binomial_datasets[Binomial_datasets['Binomial'] == 'Normal']
attack = Binomial_datasets[Binomial_datasets['Binomial'] == 'Attack']

sns.violinplot(x = normal['count'],ax = axes[0],hue = normal['Binomial'])
axes[0].set_title('Normal Distribution Based On Count')
sns.violinplot(x = attack['count'],ax = axes[1],hue = attack['Binomial'])
axes[1].set_title('Attack Distribution Based On Count')

plt.show()

###same_srv_rate 


fig,axes = plt.subplots(1,2,figsize = (10,2))
normal = Binomial_datasets[Binomial_datasets['Binomial'] == 'Normal']
attack = Binomial_datasets[Binomial_datasets['Binomial'] == 'Attack']

sns.histplot(x = normal['same_srv_rate'],ax = axes[0],hue = normal['Binomial'],kde = True,bins = 10)
axes[0].set_title('Normal Distribution Based On Same_Srv_Rate')
sns.histplot(x = attack['same_srv_rate'],ax = axes[1],hue = attack['Binomial'],kde = True,bins = 10)
axes[1].set_title('Attack Distribution Based On Same_Srv_Rate')

plt.show()


Binomial_datasets

X = Binomial_datasets.drop(columns = 'Binomial')
Y = Binomial_datasets['Binomial']

Binomial_datasets['Binomial'].value_counts()

### as data is highly imbalanced
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

X_train,X_valid, Y_train,Y_valid = train_test_split(X,Y,test_size = 0.2,random_state= 101)

smote_ = SMOTE(random_state= 102)

X_train_resampled, Y_train_resampled = smote_.fit_resample(X_train,Y_train)

print('resample_started !')
X_train_resampled.shape
Y_train_resampled.shape
print('resample_finished !')

### before removing columns the accuracy and all 

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE,SelectKBest,chi2
from sklearn.metrics import accuracy_score,classification_report,f1_score,precision_score,recall_score,confusion_matrix
k_best = SelectKBest(score_func= chi2,k = 20)
X_k_best = k_best.fit_transform(X_train_resampled,Y_train_resampled)

imp_k_best_features = X_train_resampled.columns[k_best.get_support()].tolist()

X_train_resampled = X_train_resampled[imp_k_best_features]


random_before = RandomForestClassifier(n_estimators= 50,random_state= 101)
rfe = RFE(random_before,n_features_to_select= 10)
rfe.fit(X_train_resampled,Y_train_resampled)
print(X_train_resampled.shape)

imp_features = X_train_resampled.columns[rfe.support_].tolist()
X_train_resampled = X_train_resampled[imp_features]


random_before = RandomForestClassifier(n_estimators= 200,max_depth = 10)
rand = random_before.fit(X_train_resampled,Y_train_resampled)
random_before_predict = rand.predict(X_train_resampled)

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

print(f'Train Data {accuracy_score(Y_train_resampled,random_before_predict)}')
print(f'Train Data {classification_report(Y_train_resampled,random_before_predict)}')
print(f'Train Data {confusion_matrix(Y_train_resampled,random_before_predict)}')


X_valid = X_valid[imp_features]
random_predict_test = rand.predict(X_valid)

print(f'Test Data {accuracy_score(Y_valid,random_predict_test)}')
print(f'Test Data {classification_report(Y_valid,random_predict_test)}')
print(f'Test Data {confusion_matrix(Y_valid,random_predict_test)}')



### visuals of confusion matrixs 
sns.heatmap(confusion_matrix(Y_valid,random_predict_test),annot= True,fmt = 'd',cmap = 'Blues',xticklabels= ['Attack','Normal'],yticklabels=['Attack','Normal'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion_Matrix')
plt.show()

print(imp_features)

