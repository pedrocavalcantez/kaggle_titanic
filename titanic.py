import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
#import seaborn.objects as so
from plotnine import *
from lazypredict.Supervised import LazyClassifier
from lazypredict.Supervised import LazyRegressor
sns.set()

import warnings
warnings.filterwarnings("ignore") 


#CUSTOM PRE PROCESSING

class ColumnsDrop(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns       
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X.drop(self.columns,axis=1)

#train['AgeGroup'] = pd.cut(train['Age'],5)
#print(train_data[['AgeGroup', 'Survived']].groupby('AgeGroup', as_index=False).mean().sort_values('Survived', ascending=False))


class Cut(BaseEstimator, TransformerMixin):
    def __init__(self, n_cuts,name_column):
        self.n_cuts= n_cuts
        self.name_column= name_column

    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        X_copy=X.copy()
        X_copy.cut=pd.cut(X[self.name_column],self.n_cuts)
        return X_copy
    
#000. Separation of the Columns
#columns_drop=np.array(['PassengerId'])
columns_numerical=np.array(['Pclass','Age','SibSp','Parch','Fare'])
columns_categorical=np.array(['Sex'])
x_variables=np.concatenate([columns_numerical,columns_categorical])
y_variable='Survived'


#0. Import Data
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
gender_submission=pd.read_csv('gender_submission.csv')

#1. Understanding Data
print(100*(train[y_variable].value_counts())/len(train.index))
sns.heatmap(train.drop('PassengerId',axis=1).corr(), vmax=0.6, square=True, annot=True)


variables_not_show_on_ggplot=['Fare','Age','Parch']
         
df_analysis=pd.melt(
train,
id_vars=y_variable,
var_name='column',
value_name='str_variable',
value_vars=x_variables[~(np.isin(x_variables,variables_not_show_on_ggplot))]).groupby(['str_variable',
'column',y_variable], dropna=False).size().reset_index().rename({0:'count'},axis=1)
                                                                          
df_analysis=pd.concat([df_analysis,
df_analysis.groupby(['column','str_variable']).transform(lambda x: x/sum(x)).rename({'count':'prob'},axis=1)['prob']],
axis=1).assign(str_variable=lambda x: x['str_variable'].astype(str),
bool_value=lambda x: x[y_variable].astype(bool))

ggplot(df_analysis) +aes(x="str_variable", y='prob',fill="bool_value") + geom_bar(stat='identity')+facet_wrap('column',scales='free') 




                                                          
#1. Manipulate Data
# Processing Numerical and Categorical
preprocessor = ColumnTransformer([
   # ('dropper', ColumnsDrop(columns_drop),columns_drop),
    ('median', SimpleImputer(strategy='median'), ['Fare']),
    ('numerical', SimpleImputer(strategy='constant'), columns_numerical),
   # ('cut', KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform'),['Age']),


    ('categorical', Pipeline(steps=[
    ('imputer', SimpleImputer(fill_value='NAN',strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))]),columns_categorical)])   


#ALTER TRAIN and TEST
train_altered = preprocessor.fit_transform(train.drop(y_variable,axis=1))
test_altered = pd.DataFrame(preprocessor.transform(test))
#DO a LAZY CLASSIFIER
X_train, X_test, y_train, y_test = train_test_split(train_altered, 
                                                    train[y_variable],
                                                    test_size=.5,
                                                    random_state =0)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
models


#Using the Best Model
#model=KNeighborsClassifier(n_neighbors=3)
model=LGBMClassifier(learning_rate=0.35)


#FIT and PREDICT
model.fit(train_altered,train[y_variable])
prediction=model.predict(train_altered)

sum(prediction==train[y_variable])/len(train.index)
#Analysis of Prediction
confusion_matrix(train[y_variable],prediction)




export=pd.concat([test.iloc[:,0],pd.DataFrame(model.predict(test_altered))],axis=1,ignore_index=False).rename({0:y_variable},axis=1)
export.to_csv('output.csv',index=False)

