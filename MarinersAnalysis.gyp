#%% 
##############################
### DATA/LIBRARIES READING ###  
##############################

## Import libraries and dataset 
import pandas as pd 
from tqdm import tqdm 
from pandas.api.types import is_numeric_dtype
import seaborn as sns 

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


baseball = baseball.replace({'tilt': tilt_dict})
baseballTest = baseballTest.replace({'tilt': tilt_dict})

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
                                'balls', 
                                'strikes', 
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

baseballY = baseball[['is_swing']]


## Confirm that data is consistent across train and test 
for col in baseballX: 
    if is_numeric_dtype(baseballX[col]) == False:
        trainSet = set(baseballX[col])
        testSet = set(baseballTest[col])

        res = trainSet - testSet 

        if len(res) > 0: 
            print(col)


baseballX = baseballX.dropna()


#%%
#################################
### EXPLORATORY DATA ANALYSIS ###
#################################

## Describe 
baseballX.describe().apply(lambda s: s.apply(lambda x: format(x, 'g')))

## High correlation dim reduction to follow
corrMatrix = baseballX.corr()

## Distro of variables 
baseballX['is_swing'].hist()
classBalance = len(baseballX[baseballX.is_swing == 1]) / len(baseballX)



#%%
##################
### MODEL PREP ###
##################

## Additional libraries 
from category_encoders import *


## Target Encoding 
encodedColumns = []
for col in baseballX: 
    if is_numeric_dtype(baseballX[col]) == False:
        encoder = TargetEncoder()
        colName = str(col) + '_Encoded'
        encodedColumns.append(col)
        baseballX[colName] = encoder.fit_transform(baseballX[col], baseballX['is_swing'])

## Save a full version of the dataframe
baseballXFull = baseballX

## Drop target variable, unencoded variables from training data 
baseballX = baseballX.drop(['is_swing'], axis=1)
baseballX = baseballX.drop(encodedColumns, axis=1)

## High correlation dimensionality reduction
corrMatrix = pd.DataFrame(baseballX.corr()).abs()
corrMatrix.loc['average'] = corrMatrix.mean()






#%% 
### MODEL WORK 
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

### Andrew Feedback

## EDA 
## Correlation matrix 
## Distributions 
## Compare test vs train data
## Outlier detection

## Model 
## Is black box okay, or does feature impact need to be measured 
## Linear vs non-linear 
## Accuracy vs Interperatbility graph: Intro to statistical learning 
## benchmark models 

## Feature Engineering 
## One hot encoding - turn ever level into a dummy
## Target encoding - fewer features 
## class imbalance - smote 
## Exploratory data analysis
## Dim Reduction:  50 + (8 * k) (confirm online) > rows 
## Dim Reduction: high correlation + bourtua


X_train, X_test, y_train, y_test = train_test_split(baseballX, baseballY, test_size=0.33, random_state=42)

## Grid Search CV 
# Random Grid search instead of exhaustive 

# model = RandomForestClassifier(n_estimators = 100, random_state = 24, verbose = 1)
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

forest = RandomForestClassifier()

model = GridSearchCV(forest, hyperF, cv = 3, verbose = 10, 
                      n_jobs = -1)

model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f1)



# %%
### IDEAS 
# Categorical Only model
# Continuous only model 
# Mixed model (xgboost, random forest)