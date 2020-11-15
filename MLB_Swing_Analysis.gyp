#%%
##############################
### DATA/LIBRARIES READING ###  
##############################

## Import libraries and dataset 
import pandas as pd 
from tqdm import tqdm 
from pandas.api.types import is_numeric_dtype
import seaborn as sns 
import numpy as np

pd.set_option('display.max_columns', None)
baseball = pd.read_csv('2021-train.csv')
baseballTest = pd.read_csv('2021-test.csv')

#%%
#################################
### IMPUTING/CLEANING COLUMNS ###
#################################
 
## Define Is Swing Column
baseball['is_swing'] = ((baseball.pitch_call == 'FoulBall') | (baseball.pitch_call == 'InPlay') | (baseball.pitch_call == 'StrikeSwinging')).astype('int')
baseball = baseball.dropna()


## Map Tilt variables 
tilt_dict = {'01:00:00.0000000': '01:00',
            '01:15:00.0000000': '01:15', 
            '01:30:00.0000000': '01:30', 
            '01:45:00.0000000': '01:45', 
            '02:00:00.0000000': '02:00', 
            '02:15:00.0000000': '02:15', 
            '02:30:00.0000000': '02:30', 
            '02:45:00.0000000':  '02:45',
            '03:00:00.0000000': '03:00',
            '03:15:00.0000000': '03:15',
            '03:30:00.0000000': '03:30',
            '03:45:00.0000000': '03:45',
            '04:00:00.0000000':  '04:00',
            '04:15:00.0000000':  '04:15',
            '04:30:00.0000000': '04:30',
            '04:45:00.0000000':  '04:45',
            '05:00:00.0000000':  '05:00',
            '05:15:00.0000000':  '05:15',
            '05:30:00.0000000':  '05:30',
            '05:45:00.0000000': '05:45',
            '06:00:00.0000000': '06:00',
            '06:15:00.0000000': '06:15',
            '06:30:00.0000000': '06:30',
            '06:45:00.0000000': '06:45',
            '07:00:00.0000000': '07:00',
            '07:15:00.0000000': '07:15',
            '07:30:00.0000000': '07:30',
            '07:45:00.0000000': '07:45',
            '08:00:00.0000000': '08:00',
            '08:15:00.0000000': '08:15',
            '08:30:00.0000000':  '08:30',
            '08:45:00.0000000': '08:45',
            '09:00:00.0000000':  '09:00',
            '09:15:00.0000000':  '09:15',
            '09:30:00.0000000': '09:30',
            '09:45:00.0000000':  '09:45',
            '10:00:00.0000000':  '10:00',
            '10:15:00.0000000':  '10:15',
            '10:30:00.0000000':  '10:30',
            '10:45:00.0000000':  '10:45',
            '11:00:00.0000000':  '11:00',
            '11:15:00.0000000':  '11:15',
            '11:30:00.0000000':  '11:30',
            '11:45:00.0000000':  '11:45',
            '12:00:00.0000000':  '12:00',
            '12:15:00.0000000':  '12:15',
            '12:30:00.0000000':  '12:30',
            '12:45:00.0000000':  '12:45',
            '1:00': '01:00',
            '1:15': '01:15',
            '1:30': '01:30',
            '1:45': '01:45',
            '2:00': '02:00',
            '2:15': '02:15',
            '2:30': '02:30',
            '2:45': '02:45',
            '3:00': '03:00',
            '3:15': '03:15',
            '3:30': '03:30',
            '3:45': '03:45',
            '4:00': '04:00',
            '4:15': '04:15',
            '4:30': '04:30',
            '4:45': '04:45',
            '5:00': '05:00',
            '5:15': '05:15',
            '5:30': '05:30',
            '5:45': '05:45',
            '6:00': '06:00',
            '6:15': '06:15',
            '6:30': '06:30',
            '6:45': '06:45',
            '7:00': '07:00',
            '7:15': '07:15',
            '7:30': '07:30',
            '7:45': '07:45',
            '8:00': '08:00',
            '8:15': '08:15',
            '8:30': '08:30',
            '8:45': '08:45',
            '9:00': '09:00',
            '9:15': '09:15',
            '9:30': '09:30',
            '9:45': '09:45'}

## Replace columns with mapped values and correct data types
## Define 'Count' column
baseball = baseball.replace({'tilt': tilt_dict})
baseball['Count'] = (baseball.balls).astype('str') + (baseball.strikes).astype('str')

baseballTest = baseballTest.replace({'tilt': tilt_dict})
baseballTest['Count'] = (baseball.balls).astype('str') + (baseball.strikes).astype('str')

## Drop null values
baseball = baseball.dropna()

## Remoe unhelpful variables and dependent variable root (pitch call)
baseballX = baseball[[ 'level',
                                'pitcher_id', ## cut?
                                'pitcher_side',
                                'batter_id', ## cut     
                                'batter_side', 
                                'stadium_id', ## cut
                                'umpire_id', ## cut
                                'catcher_id', ## cut
                                'inning', 
                                'top_bottom', ## cut
                                'outs', 
                                #'balls', 
                                #'strikes', 
                                'Count', 
                                'release_speed', 
                                'vert_release_angle', 
                                'horz_release_angle', 
                                'spin_rate', 
                                'spin_axis', 
                                'tilt', 
                                'rel_height', 
                                'rel_side', 
                                'extension', 
                                'vert_break', 
                                'induced_vert_break', 
                                'horz_break', 
                                'plate_height', 
                                'plate_side', 
                                'zone_speed', 
                                'vert_approach_angle', 
                                'horz_approach_angle', 
                                'x55', 
                                # 'y55', 
                                'z55', 
                                'pitch_type', 
                                ## 'pitch_call', 
                                'is_swing'
                                ]]

## Define dependent variable
baseballY = baseball[['is_swing']]


## Confirm that data is consistent across train and test 
for col in baseballX: 
    if is_numeric_dtype(baseballX[col]) == False:
        trainSet = set(baseballX[col])
        testSet = set(baseballTest[col])

        res = trainSet - testSet 

        if len(res) > 0: 
            print(col)




#%%
#################################
### EXPLORATORY DATA ANALYSIS ###
#################################
from scipy import stats 
## Describe 
baseballX.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

## High correlation dim reduction to follow
corrMatrix = baseballX.corr()

## Distribution of variables 
baseballX['is_swing'].hist()
classBalance = len(baseballX[baseballX.is_swing == 1]) / len(baseballX)

## Outliers - None 
len(baseballX[(np.abs(stats.zscore(baseballX)) > 1).all(axis=1)]) 

#%%
##################
### MODEL PREP ###
##################

## Additional libraries 
from category_encoders import *
from collections import Counter
from sklearn.decomposition import PCA
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer


## Target Encoding 
encodedColumns = []
value_dicts = []
for col in baseballX: 
    if is_numeric_dtype(baseballX[col]) == False:
        encoder = TargetEncoder()
        colName = str(col) + '_Encoded'
        encodedColumns.append(col)
        baseballX[colName] = encoder.fit_transform(baseballX[col], baseballX['is_swing'])

        value_dicts.append(pd.Series(baseballX[colName].values,index=baseballX[col]).to_dict())
        print(colName)



# ## One Hot Encoding 
# ## This method was deferred in favor of Target Encoding
# encodedColumns = []
# for col in baseballX: 
#     if is_numeric_dtype(baseballX[col]) == False:
#         print(col)
#         encoder = LabelBinarizer()
#         encoder.fit(baseballX[col])
#         transformed = encoder.transform(baseballX[col])
#         ohe_df = pd.DataFrame(transformed)
#         ohe_df.columns = encoder.classes_
#         encodedColumns.append(col)
#         baseballX = baseballX.join(ohe_df)

## Save a full version of the dataframe in case
baseballXFull = baseballX

## Drop target variable, unencoded variables from training data 
baseballX = baseballX.drop(['is_swing'], axis=1)
baseballX = baseballX.drop(encodedColumns, axis=1)

## High correlation dimensionality reduction
corrMatrix = pd.DataFrame(baseballX.corr()).abs()
corrMatrix.loc['average'] = corrMatrix.mean()

high_corrs = []
for idx,row in corrMatrix.iterrows(): 
    for col in corrMatrix.columns: 
        if (row[col] < 1) & (row[col] > 0.8): 
            high_corrs.append(idx)
            high_corrs.append(col)

print(Counter(high_corrs))

## Removing the following for high correlation
# Consolidating related variables into one variable using PCA
pca = PCA(n_components=1)
hor_location = pca.fit_transform(baseballX[['x55', 'horz_release_angle', 'rel_side', 'pitcher_side_Encoded']])
pca.explained_variance_ratio_

pca = PCA(n_components=1)
ver_location = pca.fit_transform(baseballX[['release_speed', 'vert_break', 'zone_speed', 'induced_vert_break']])
pca.explained_variance_ratio_

## Between release height and Z55, we take release height (lower average variation)
list(set(high_corrs)).remove('rel_height')
baseballX = baseballX.drop(high_corrs, axis=1)
baseballX['ver_location'] = ver_location
baseballX['hor_location'] = hor_location

print(baseballX.columns)

#%%
#################################
### Boruta Feature Extraction ###
#################################

### Initialize Boruta
import random
forest = RandomForestRegressor(
   n_jobs = -1, 
   max_depth = 7, 
   verbose = 2
)
boruta = BorutaPy(
   estimator = forest, 
   n_estimators = 'auto',
   max_iter = 50, # number of trials to perform, 
   verbose = 2
)
### fit Boruta (it accepts np.array, not pd.DataFrame)
row_samples = []
xList = [1] + ([0] * 99)
for i in range(len(baseballX)):
    row_samples.append(random.choice(xList))

## Define sample indices, calculate percentage of original dataset included in sample
bool_list = list(map(bool,row_samples))
bool_list
print(sum(bool_list) / len(bool_list))

baseballXSample = baseballX[np.array(bool_list).astype(bool)]
baseballYSample = baseballY[np.array(bool_list).astype(bool)]

#%%
##################
### RUN BORUTA ###
##################

boruta.fit(np.array(baseballXSample), np.array(baseballYSample))
### print results
green_area = baseballXSample.columns[boruta.support_].to_list()
blue_area = baseballXSample.columns[boruta.support_weak_].to_list()
print('features in the green area:', green_area)
print('features in the blue area:', blue_area)

#%%
##########################################
### Feature Extraction based on Boruta ###
########################################## 

## Shortcut for repeated runs to avoid having to run boruta each time
# green_area = ['vert_release_angle', 'spin_rate', 'spin_axis',
#        'horz_break', 'plate_height', 'plate_side', 'vert_approach_angle',
#        'pitcher_id_Encoded', 'batter_id_Encoded', 'batter_side_Encoded',
#        'tilt_Encoded', 'ver_location', 'hor_location', 'Count_Encoded']

## Subset on columns to keep 
baseballX = baseballX[green_area]

## Report the remaining columns
print(baseballX.columns)

## Describe 
baseballXSummary = baseballX.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

## New correlation looks a lot better
corrMatrix = baseballX.corr()

## Confirm that IVs and DVs match up
print('Number of IV Records :', len(baseballX), ' Number of DV Records :', len(baseballY))


#%% 
##################################
### MODEL WORK - RANDOM FOREST ###
################################## 
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(baseballX, baseballY, test_size=0.25, random_state=1)
## In case user wants to run model on sample
# X_train, X_test, y_train, y_test = train_test_split(baseballXSample, baseballYSample, test_size=0.25, random_state=42)

# Random Grid search instead of exhaustive for computational expense reasons

# model = RandomForestClassifier(n_estimators = 100, random_state = 24, verbose = 1)
n_estimators = [1500]
max_depth = [15, 20, 25, 30]
min_samples_split = [75, 100, 125, 150]
min_samples_leaf = [8, 9, 10, 11, 12] 
max_features = [11, 12, 13, 14]
warm_start = [True, False]

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf, max_features = max_features, warm_start = warm_start)

## Save the best parameters for repeated runs
# forest = RandomForestClassifier(verbose = 10, max_depth = 15, 
#                                   max_features = 13, min_samples_leaf = 11, 
#                                   min_samples_split= 75, n_estimators= 1500, 
#                                   warm_start = True, n_jobs = -1)

forest = RandomForestClassifier()
model = RandomizedSearchCV(forest, hyperF, cv = 5, verbose = 100, 
                       n_jobs = -1, n_iter = 50)

# model = forest

## Fit and run model
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print('F1: ',f1)
print('Accuracy: ', acc)

#%%
## Test Predictions
for col in ['pitcher_id', 'batter_id', 'batter_side', 'Count', 'tilt', 'pitcher_side']: 
    idx = encodedColumns.index(col)
    colDict = value_dicts[idx]
    baseballTest = baseballTest.replace({col: colDict})
    colName = str(col) + '_Encoded'

    ## If the value has not been mapped, then replace the value with the column mean 
    ## Replacing with column mean should give a value that is amgbigious regarding prediction power/direction
    colMean = baseballTest.loc[baseballTest[col].apply(lambda x: isinstance(x, float)), col].mean()
    baseballTest.loc[baseballTest[col].apply(lambda x: isinstance(x, str)), col] = colMean

    baseballTest[colName] = baseballTest[col]

    print(col)


# Consolidating related variables into one variable 
pca = PCA(n_components=1)
hor_location = pca.fit_transform(baseballTest[['x55', 'horz_release_angle', 'rel_side', 'pitcher_side_Encoded']])
pca.explained_variance_ratio_

pca = PCA(n_components=1)
ver_location = pca.fit_transform(baseballTest[['release_speed', 'vert_break', 'zone_speed', 'induced_vert_break']])
pca.explained_variance_ratio_

baseballTest['ver_location'] = ver_location
baseballTest['hor_location'] = hor_location

#%%
## Run model and predict
baseballTest = baseballTest[baseballX.columns]

for col in tqdm(baseballTest.columns):
    baseballTest[col] = baseballTest[col].fillna(baseballTest[col].mean())

y_pred = model.predict(baseballTest)
print(y_pred)


## Get original pitch IDs
## Thankfully, there has been no re-sorting of the test data frame 
baseballTest2 = pd.read_csv('2021-test.csv')
predictions = pd.DataFrame(y_pred, columns = ['is_swing'])
predictions['pitch_id'] = baseballTest2['pitch_id']

predictions.to_csv('swingPredictions.csv')

#%%
#######################
### DEPRECATED CODE ###
#######################

# Any code beyond this point is from previous iterating and ideating
# Some of the code was used to confirm assumptions noted in the write up 

#%%
########################################
### MODEL WORK - LOGISTIC REGRESSION ###
########################################
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


pipe = Pipeline([('classifier' , RandomForestClassifier(verbose = 10))])

# Create param grid
param_grid = [
    {'classifier' : [LogisticRegression()],
     'classifier__penalty' : ['l1', 'l2'],
    'classifier__solver' : ['liblinear']},
]

model = GridSearchCV(pipe, param_grid = param_grid, cv = 5, verbose=100, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f1)
print('Logit: ', model.best_estimator_)

#%%
########################
### MODEL WORK - SVM ###
########################
from sklearn.svm import LinearSVC
from sklearn import svm
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from matplotlib import pyplot as plt

# X_train, X_test, y_train, y_test = train_test_split(baseballX, baseballY, test_size=0.33, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(baseballXSample, baseballYSample, test_size=0.25, random_state=42)


## Create grid search space, conduct grid search, and fit model
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
model = RandomizedSearchCV(svm.SVC(verbose = 200),param_grid, n_jobs = -1, cv =5, n_iter = 30, verbose=200)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f1)

print('SVM: ', model.best_estimator_)
