import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer,KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import math


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

finalized_table.head()

### merged the data 
finalized_table.to_csv('finalized_table.csv')

## No missings
print(finalized_table.isna().sum())

print('Finished !')

## No Duplications 
print(finalized_table[finalized_table.duplicated()]) 

## for binomial 

finalized_table['Binomial'] = np.where(finalized_table['Attack'] == 'normal','Normal','Attack')

### dividing x and y 

X = finalized_table.drop(columns = ['Attack','Binomial'])
Y = finalized_table['Binomial']

### Outlier Detection 

X.describe(percentiles= [.25,.55,.99,.999,.9999])

for col in X.columns:
    X[col] = X[col].clip(lower= X[col].quantile(0.01),upper= X[col].quantile(0.99))


X.describe(percentiles= [.25,.55,.99,.999,.9999])

count = 1
indexes = []

while count <= X.shape[0]:
    indexes.append(count)
    count+=1

X['indexs'] = indexes


### variance Inflation Factor (to get imp columns only)


p_valuess = pd.DataFrame({'cols': X.columns ,'vif_values' :[variance_inflation_factor(X.values,col) for col in range(len(X.columns))]})

print(p_valuess)

features = [i for i in p_valuess['cols'] if i != ' flag']
features

p_valuess['cols'] = features

extracted_features = []

extracted_features.extend(p_valuess[p_valuess['vif_values']<700]['cols'])

X = X[X.columns[X.columns.isin(extracted_features)]]
Y
print(type(Y))

X.shape[1]

### now no outliers 

colss = 4
rows = math.ceil(X.shape[1]/colss)
fig,axes = plt.subplots(rows,colss)
axes = axes.flatten()

for i in X.columns:
    grps = X.groupby(i)['indexs'].count().reset_index(name = 'counts')
    axes[i].boxplot(grps[i],grps['counts'])
    axes[i].set_title(i)
    
plt.tight_layout() 
plt.show()

### Visuals 

X
Y_ = Y

### for correlation encoding the y variables 

labels = LabelEncoder()
Y_ = labels.fit_transform(Y)

Y_

Binary_datasets = pd.concat([X,Y],axis = 1)

Binary_datasets

### first distribution of binomial

dist_binomial = Binary_datasets.groupby('Binomial')['indexs'].count().reset_index(name = 'counts')
dist_binomial['counts'] = (dist_binomial['counts']*100/dist_binomial['counts'].sum())
m = sns.barplot(data = dist_binomial, x = 'Binomial', y = 'counts')
for i  in m.patches:
    plt.text((i.get_x() + i.get_width() / 2),i.get_height(),str(f'{i.get_height():.0f}%'),ha = 'center', va = 'bottom' )
plt.title('Binomial Distributions % ')
plt.show()

#Y = Y.tolist()
print(type(Y))


Y  = pd.DataFrame(data = Y, columns = ['Binomial'])

Binary_datasets_corr = pd.concat([X,Y],axis = 1)

cols_selected_ = [col for col in Binary_datasets_corr.columns if  col!= 'indexs']

#col != 'Binomial' and

corr_matrix = Binary_datasets_corr[cols_selected_].corr()
sns.heatmap(corr_matrix[['Binomial']],annot = True)
plt.show()

corr_matrix_binomials = corr_matrix[['Binomial']]

attack = Binary_datasets[Binary_datasets['Binomial'] == 'Attack']
normal = Binary_datasets[Binary_datasets['Binomial'] == 'Normal']

Binary_datasets.groupby('Binomial')['duration'].describe()

### dist of binomial based on duration 
fig,axes = plt.subplots(2,1,figsize = (10,5))
sns.histplot(normal['duration'],bins = 10,kde = True,ax = axes[0])
axes[0].set_title('Normal Distributions Based On Duration')

sns.histplot(attack['duration'],bins = 10,kde = True,ax = axes[1])
axes[1].set_title('Attack Distributions Based On Duration')

plt.tight_layout()
plt.show()


### based on logged in

dist_logged_in = Binary_datasets.groupby('Binomial')[' logged_in'].value_counts(normalize= True).reset_index(name = "dist_prcnt")

sns.barplot(dist_logged_in,x = dist_logged_in['Binomial'],y = dist_logged_in['dist_prcnt'],hue = dist_logged_in[' logged_in'])
plt.xlabel(dist_logged_in['Binomial'])
plt.ylabel("% logged_in ")
plt.legend(loc = 'upper left')
plt.title('Distribution Of Logged_In Based On Binomail')
plt.show()

# in normal logged in is more proportions and in case of attack logged_in about 0.99 getting failed
# 1 if successfully logged in; 0 otherwise 

Binary_datasets[' flag'].unique()




