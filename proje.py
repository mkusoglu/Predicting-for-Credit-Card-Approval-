import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,recall_score,roc_auc_score,f1_score,plot_confusion_matrix,plot_roc_curve,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

fig=plt.gcf()
fig.set_size_inches(12,6)

df = pd.read_csv("application_record.csv", encoding = 'utf-8') 
record = pd.read_csv("credit_record.csv", encoding = 'utf-8') 

#df sayısal verilerde düzenleme
df['CODE_GENDER'] = df['CODE_GENDER'].replace(['M','F'],[0,1])#ERKEK=0,KADIN=1
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].replace(['N','Y'],[0,1])
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].replace(['N','Y'],[0,1])
#yaşını bulma
df['DAYS_BIRTH']=df['DAYS_BIRTH']*(-1)
df['DAYS_BIRTH']=np.round(df['DAYS_BIRTH']/365)

#çalışıyor mu (burası daha farklı şekilde manipüle edilebilir.)
df['DAYS_EMPLOYED']=np.round(df['DAYS_EMPLOYED']/365,2)
#tek pozitif sayı 1000.67, rakam pozitif ise çalışmıyor demek. çalışmayanlara 0 atandı.
df['DAYS_EMPLOYED']=df['DAYS_EMPLOYED']*(-1)
df['DAYS_EMPLOYED'][df['DAYS_EMPLOYED'] == -1000.67]=0

df.NAME_INCOME_TYPE.value_counts()/df.shape[0]
df.NAME_EDUCATION_TYPE.value_counts()
df.NAME_FAMILY_STATUS.value_counts()
df.NAME_HOUSING_TYPE.value_counts()
df.OCCUPATION_TYPE.value_counts()
#null count
df.NAME_INCOME_TYPE.isna().sum() 
df.NAME_EDUCATION_TYPE.isna().sum() 
df.NAME_FAMILY_STATUS.isna().sum() 
df.NAME_HOUSING_TYPE.isna().sum() 
df.OCCUPATION_TYPE.isna().sum() 
df.OCCUPATION_TYPE.isna().sum() #sadece bunda veri var

df.OCCUPATION_TYPE.isna().sum()/df.shape[0]#yüzde kaç
#bunlar eksik veri değiller. Mesela çoğu emekli nan
df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].replace(np.nan, 'Other')

record['STATUS'][record['STATUS'] =='5']='7'
record['STATUS'][record['STATUS'] =='4']='6' 
record['STATUS'][record['STATUS'] =='3']='5'
record['STATUS'][record['STATUS'] =='2']='4'
record['STATUS'][record['STATUS'] =='1']='3' 
record['STATUS'][record['STATUS'] =='0']='2' 
record['STATUS'][record['STATUS'] =='X']='0' 
record['STATUS'][record['STATUS'] =='C']='1' 

begin_month=pd.DataFrame(record.groupby(["ID"])["MONTHS_BALANCE"].agg(min))
begin_month['MONTHS_BALANCE']=begin_month['MONTHS_BALANCE']*(-1)

customer_max_statu=pd.DataFrame(record.groupby(["ID"])["STATUS"].agg(max)) 

customer_statu_count = record.groupby(['ID', 'STATUS']).size().reset_index(name ='max_statu_count')
customer_max_statu_count=pd.DataFrame(customer_statu_count.groupby(["ID"])["max_statu_count"].agg(max)) 

new_record=pd.merge(begin_month,customer_max_statu,how="left",on="ID")
new_record=pd.merge(new_record,customer_max_statu_count,how="left",on="ID")

merged=pd.merge(df,new_record,how="inner",on="ID")

merged.STATUS.value_counts()

# aykırı değerleri çıkarmaca
other_numerical_cols = ["Income","Age","Experience","Household_Size"]

"""
fig = make_subplots(rows=2, cols=2, start_cell="bottom-left",
                   subplot_titles=("CNT_CHILDREN", "DAYS_EMPLOYED", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"))

fig.add_trace(go.Box(x=df.CNT_CHILDREN, name='CNT_CHILDREN',boxmean=True),row=1,col=1)
fig.add_trace(go.Box(x=df.DAYS_EMPLOYED, name='DAYS_EMPLOYED', boxmean=True), row=1, col=2)
fig.add_trace(go.Box(x=df.AMT_INCOME_TOTAL, name='AMT_INCOME_TOTAL', boxmean=True), row=2, col=1)
fig.add_trace(go.Box(x=df.CNT_FAM_MEMBERS, name="CNT_FAM_MEMBERS", boxmean=True),row=2, col=2)

fig.show()
"""
#3 ten fazla çocuğu olanlar aykırı değer, bunlar çıkarıldı
merged = merged[merged.CNT_CHILDREN < 3]

#hanedeki insan sayısı aykırı olanlar çıkarıldı.
iqr = (3.000000 - 2.000000)
upper = 3.000000 + (iqr*1.5)
merged = merged[merged.CNT_FAM_MEMBERS < upper]   

#gelirden aykırı çıkar
from scipy import stats
gelir_z=stats.zscore(merged['AMT_INCOME_TOTAL'])
merged=merged[stats.zscore(merged.AMT_INCOME_TOTAL) < 3]

#yaştan aykırıları çıkar
yaş_z=stats.zscore(merged['DAYS_BIRTH'])

#deneyimden aykırıları çıkar
deneyim_z=stats.zscore(merged['DAYS_EMPLOYED'])
deneyim_z[deneyim_z > 3].count()
merged=merged[stats.zscore(merged.DAYS_EMPLOYED) < 3]

#betimleyici anlatım
merged['NAME_INCOME_TYPE'].hist()
merged['NAME_INCOME_TYPE'].value_counts()
merged['NAME_INCOME_TYPE'].value_counts()/ merged.shape[0]

merged['NAME_FAMILY_STATUS'].hist()
merged['NAME_FAMILY_STATUS'].value_counts()
merged['NAME_FAMILY_STATUS'].value_counts()/ merged.shape[0]

merged['NAME_HOUSING_TYPE'].hist()
merged['NAME_HOUSING_TYPE'].value_counts()
merged['NAME_HOUSING_TYPE'].value_counts()/ merged.shape[0]

merged['NAME_EDUCATION_TYPE'].hist()
merged['NAME_EDUCATION_TYPE'].value_counts()
merged['NAME_EDUCATION_TYPE'].value_counts()/ merged.shape[0]

fig, axes = plt.subplots(1,2)

g1=sns.countplot(y=merged.NAME_INCOME_TYPE,linewidth=1.2, ax=axes[0])
g1.set_title("Customer Distribution by Income Type")
g1.set_xlabel("Count")

g2=sns.countplot(y=merged.NAME_FAMILY_STATUS,linewidth=1.2, ax=axes[1])
g2.set_title("Customer Distribution by Family Status")
g2.set_xlabel("Count")

fig.set_size_inches(14,5)

plt.tight_layout()


plt.show()

fig, axes = plt.subplots(1,2)

g1= sns.countplot(y=merged.NAME_HOUSING_TYPE,linewidth=1.2, ax=axes[0])
g1.set_title("Customer Distribution by Housing Type")
g1.set_xlabel("Count")
g1.set_ylabel("Housing Type")

g2= sns.countplot(y=merged.NAME_EDUCATION_TYPE, ax=axes[1])
g2.set_title("Customer Distribution by Education")
g2.set_xlabel("Count")
g2.set_ylabel("Education Type")

fig.set_size_inches(14,5)

plt.tight_layout()

plt.show()



#ilk model denemeleri
merged_model=merged[['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'DAYS_BIRTH',
       'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
       'FLAG_EMAIL', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'MONTHS_BALANCE',
       'STATUS', 'max_statu_count']]


merged_model['behav'] = None
merged_model['behav'][merged_model['STATUS'] =='0']='Good' 
merged_model['behav'][merged_model['STATUS'] =='1']='Good' 
merged_model['behav'][merged_model['STATUS'] =='2']='Good' 
merged_model['behav'][merged_model['STATUS'] =='3']='Good' 
merged_model['behav'][merged_model['STATUS'] =='4']='Bad' 
merged_model['behav'][merged_model['STATUS'] =='5']='Bad' 
merged_model['behav'][merged_model['STATUS'] =='6']='Bad' 
merged_model['behav'][merged_model['STATUS'] =='7']='Bad' 
merged_model['behav'] = merged_model['behav'].replace(['Bad','Good'],[0,1])

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
merged_model['AMT_INCOME_TOTAL']=min_max_scaler.fit_transform(merged_model[['AMT_INCOME_TOTAL']])
#merged_model['DAYS_BIRTH']=min_max_scaler.fit_transform(merged_model[['DAYS_BIRTH']])
merged_model['DAYS_EMPLOYED']=min_max_scaler.fit_transform(merged_model[['DAYS_EMPLOYED']])
merged_model['CNT_FAM_MEMBERS']=min_max_scaler.fit_transform(merged_model[['CNT_FAM_MEMBERS']])
merged_model['MONTHS_BALANCE']=min_max_scaler.fit_transform(merged_model[['MONTHS_BALANCE']])
merged_model['STATUS']=min_max_scaler.fit_transform(merged_model[['STATUS']])
merged_model['max_statu_count']=min_max_scaler.fit_transform(merged_model[['max_statu_count']])
merged_dum=pd.get_dummies(merged_model)

column_to_move = merged_dum.pop("behav")


merged_dum.insert(42, "behav", column_to_move)
X=merged_dum.iloc[:,0:42]
Y=merged_dum.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.4, random_state=31)

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_balanced, y_balanced = oversample.fit_resample(x_train, y_train)
X_test_balanced, y_test_balanced = oversample.fit_resample(x_test, y_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
rfc=RandomForestClassifier(n_estimators=125)
rfc.fit(X_balanced,y_balanced)

y_pred= rfc.predict(X_test_balanced)
cm=confusion_matrix(y_test_balanced,y_pred)

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
print(cm)

accuracy_score(y_test_balanced, y_pred)

#k-means
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(merged_dum)
merged_dum['kmean']=kmeans.labels_
plt.scatter(x=merged_dum.NAME_EDUCATION_TYPE,y=merged_dum.AMT_INCOME_TOTAL,c=merged_dum['kmean'])



merged_dum=pd.get_dummies(merged)


#--------------------------
classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighbors" : KNeighborsClassifier(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=16),
    "XGBoost" : XGBClassifier(max_depth=12,
                              n_estimators=250,
                              min_child_weight=8, 
                              subsample=0.8, 
                              learning_rate =0.02,    
                              seed=42),
    "CatBoost" : CatBoostClassifier(iterations=250,
                           learning_rate=0.2,
                           od_type='Iter',
                           verbose=25,
                           depth=16,
                           random_seed=42)
}

result_table = pd.DataFrame(columns=['classifiers','accuracy','presicion','recall','f1_score','fpr','tpr','auc'])

y_test_balanced = y_test_balanced.astype(int)


classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighbors" : KNeighborsClassifier(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier(n_estimators=250,max_depth=12,min_samples_leaf=16),
    "XGBoost" : XGBClassifier(max_depth=12,
                              n_estimators=250,
                              min_child_weight=8, 
                              subsample=0.8, 
                              learning_rate =0.02,    
                              seed=42),
    "CatBoost" : CatBoostClassifier(iterations=250,
                           learning_rate=0.2,
                           od_type='Iter',
                           verbose=25,
                           depth=16,
                           random_seed=42)
}

result_table = pd.DataFrame(columns=['classifiers','accuracy','presicion','recall','f1_score','fpr','tpr','auc'])

y_test = y_test.astype(int)


for key, classifier in classifiers.items():
    classifier.fit(X_balanced, y_balanced)
    y_predict = classifier.predict(x_test)
    
    yproba = classifier.predict_proba(x_test)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    conf_matrix = confusion_matrix(y_test,y_predict)
    
    result_table = result_table.append({'classifiers':key,
                                        'accuracy':accuracy_score(y_test, y_predict),
                                        'presicion':precision_score(y_test, y_predict, average='weighted'),
                                        'recall':recall_score(y_test, y_predict, average='weighted'),
                                        'f1_score':f1_score(y_test, y_predict, average='weighted'),
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc
                                         }, ignore_index=True)
        
result_table.set_index('classifiers', inplace=True)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))

for cls, ax in zip(list(classifiers.values()), axes.flatten()):
    plot_confusion_matrix(cls, 
                          x_test, 
                          y_test, 
                          ax=ax, 
                          cmap='Blues')
    ax.title.set_text(type(cls).__name__)
plt.tight_layout()  
plt.show()


result_table.iloc[:,:4]
#***************************
