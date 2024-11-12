import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from statsmodels.stats.power import tt_ind_solve_power
import random

#load raw datafiles into dataframe and do initial data cleaning
att_current = pd.read_csv('data/sample_attendance.csv') #read raw historical attednace file
att_current['school_year'] = 'SY23-24'
att_hist = pd.read_csv('data/sample_attendance_historical.csv') #read raw current attednace file
att_hist['school_year'] = 'SY24-25'
att_combined = pd.concat([att_current, att_hist]) #create unioned dataframe of the two files
att_combined = att_combined.drop_duplicates() #drop duplicates
att_combined['reference_dt'] = pd.to_datetime(att_combined['reference_dt']) #convert string date to datetime
att_combined = att_combined.sort_values(by=['student_id', 'reference_dt']) # sort dataframe by date
att_combined['day_count'] = 1

#translate attendance to numeric codes // align values to chronic absenteeism definition in Illinois
att_combined['days_attended'] = att_combined['status'].map(
    {'present': 1,
    'tardy': 1,
    'absent': 0,
    'absentExcused': 0,
    'halfdayExcused': 0.5,
    'disciplinary': 0,
    'halfday': 0.5})

att_combined['days_missed'] = 1 - att_combined['days_attended'] # calculation for number of days missed


att_combined['cumulative_days'] = att_combined.groupby(['school_year','student_id']).cumcount() + 1 #rolling count of number of instructional days
att_combined['cumulative_days_missed'] = att_combined.groupby(['school_year','student_id'])['days_missed'].cumsum() #rolling sum of  cumulative days missed by student and school year
att_combined['cumulative_days_attended'] = att_combined.groupby(['school_year','student_id'])['days_attended'].cumsum()#rolling sum of  cumulative days attended by student and school year

att_combined['percent_absent'] = att_combined['cumulative_days_missed'] / att_combined['cumulative_days']

def moderately_absent (row):
   if row['percent_absent'] > 0.1 and row['percent_absent'] <= 0.2:
      return 1
   return 0

att_combined['moderately_absent'] = att_combined.apply(moderately_absent, axis=1)
