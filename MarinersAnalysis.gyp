#%% 
### DATA PREP 
## Import libraries and dataset 
import pandas as pd 
from tqdm import tqdm 
pd.set_option('display.max_columns', None)

pd.set_option('display.max_columns', None)
baseball = pd.read_csv('2021-train.csv')

## Define Is Swing Column
baseball['is_swing'] = ((baseball.pitch_call == 'FoulBall') | (baseball.pitch_call == 'InPlay') | (baseball.pitch_call == 'StrikeSwinging')).astype('int')

## Define dummy variables 
# Level 
baseball['A'] = (baseball.level == 'A').astype('int')
baseball['A+'] = (baseball.level == 'A+').astype('int')
baseball['AA'] = (baseball.level == 'AA').astype('int')
baseball['AAA'] = (baseball.level == 'AAA').astype('int')
baseball['MLB'] = (baseball.level == 'MLB').astype('int')

# Pitcher Side
baseball['Pitcher_Left'] = (baseball.pitcher_side == 'Left').astype('int')
baseball['Pitcher_Right'] = (baseball.pitcher_side == 'Right').astype('int')
baseball['Pitcher_S'] = (baseball.pitcher_side == 'S').astype('int')

# Batter
baseball['Batter_Left'] = (baseball.batter_side == 'Left').astype('int')
baseball['Batter_Right'] = (baseball.batter_side == 'Right').astype('int')
baseball['Batter_S'] = (baseball.batter_side == 'S').astype('int')

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


#%% 
### MODEL WORK 
baseball.head()

# %%