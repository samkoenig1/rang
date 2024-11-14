import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

#load raw datafiles into dataframe and fomrat, initial data cleaning, drop duplicates, assign school year field, and union back together
att_current = pd.read_csv('data/sample_attendance.csv')
att_current['school_year'] = 'SY23-24'
att_hist = pd.read_csv('data/sample_attendance_historical.csv')
att_hist['school_year'] = 'SY24-25'
attendance = pd.concat([att_current, att_hist])
attendance = attendance.drop_duplicates()
attendance['reference_dt'] = pd.to_datetime(attendance['reference_dt'])
attendance = attendance.sort_values(by=['student_id', 'reference_dt'])

#Add columns relevant for later analysis
attendance['days_attended'] = attendance['status'].map({'present': 1,'tardy': 1,'absent': 0,'absentExcused': 0,'halfdayExcused': 0.5,'disciplinary': 0,'halfday': 0.5}) #numerical version of days attended
attendance['days_missed'] = 1 - attendance['days_attended'] # calculation for number of days missed
attendance['cumulative_days'] = attendance.groupby(['school_year','student_id']).cumcount() + 1 #rolling count of number of instructional days
attendance['cumulative_days_missed'] = attendance.groupby(['school_year','student_id'])['days_missed'].cumsum() #rolling sum of  cumulative days missed by student and school year
attendance['cumulative_days_attended'] = attendance.groupby(['school_year','student_id'])['days_attended'].cumsum() #rolling sum of  cumulative days attended by student and school year
attendance['percent_absent'] = attendance['cumulative_days_missed'] / attendance['cumulative_days'] #Calculate hte percentage of days missed at the current point in the school year
#define treatment group for tier 1 and 2 students we want to send push notifications
def groups_experiment_one (row):
   if row['percent_absent'] >= 0.08 and row['percent_absent'] < 0.15:
      return 1
   return 0

#add groups for treatment groups. These are what we need to pass through to the ols_regression function below
attendance['experiment_one_groups'] = attendance.apply(groups_experiment_one, axis=1) #define experiment one based on chronic absenteeism rates
attendance['experiment_two_groups'] = np.random.randint(0, 2, size=len(attendance))

#Start Code for OLS Regression and Visuals.
def ols_regression(treatment_group, intervention_date, comparison_start_date, days_of_experiment):
    #Create timeseries dataframe of attendance percentage by date and treatment group, clean
    df = attendance.groupby(['reference_dt',treatment_group]).agg({'days_attended':'sum','student_id':'count'}).reset_index()
    df.rename(columns={'student_id': 'n_students'}, inplace=True) #rename columns for easier comprehension
    df['att_percent'] =  df['days_attended'] / df['n_students'] #calculate the average percentage by day
    intervention_date = pd.to_datetime(intervention_date) #convert to date time
    comparison_start_date = pd.to_datetime(comparison_start_date) #convert to date time

    #add in binary column for whether intervention is before or after
    def before_after (row):
       if row['reference_dt'] >= intervention_date:
          return 1
       return 0
    df['before_after'] = df.apply(before_after, axis=1) #add in treatement group for use when determining

    #create dataframe for the comparison and intervention period. Then union the two
    dfcomparison = df[(df['reference_dt'] >= comparison_start_date) & (df['reference_dt'] <= comparison_start_date + pd.Timedelta(days = days_of_experiment))]
    dfintervention = df[(df['reference_dt'] >= intervention_date) & (df['reference_dt'] <= intervention_date + pd.Timedelta(days = days_of_experiment))]
    df = pd.concat([dfcomparison, dfintervention])

    #add fields to make visuals clearer
    df['before_after_string'] = df['before_after'].map({0:'Before Intervention',1: 'Post Intervention'})
    df['treatment_string'] = df[treatment_group].map({0:'Control',1: 'Treatment'})

    ##### Visualize changes as box plots to check for outliers
    sns.boxplot(x='before_after_string', y='att_percent',hue= 'treatment_string', data=df,palette='husl')
    plt.title('Boxplot of Attendance Percentage by Group (Before vs Post Intervention)')
    plt.xlabel('Before / After Intervention')
    plt.ylabel('Attendance Percentage')
    plt.legend(title=treatment_group)
    plt.savefig('output/' + treatment_group + '_boxplot.png') #save file

    # Create histogram and save to folder to check for normal distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df,x='att_percent',hue='before_after_string',element='step',bins=15,palette='husl',alpha=0.5)
    plt.title('Overlaid Histograms of Attendance Percentage (Before vs After Intervention)')
    plt.xlabel('Attendance Percentage')
    plt.savefig('output/' + treatment_group + '_histogram.png')

    #Run Linear Regression using statsmodels
    df['interaction'] = df[treatment_group] * df['before_after'] # Create interaction term for treatment group and time
    x = df[[treatment_group, 'before_after', 'interaction']]
    x = sm.add_constant(x)
    y = df['att_percent']
    model = sm.OLS(y, x).fit()
    print(model.summary()) #print to terminal

    #calculate descriptive statistics and print
    grouped_stats = df.groupby(['treatment_string', 'before_after_string'])['att_percent'].describe().reset_index()
    df.to_csv('output/'+treatment_group+'_rawdata.csv') #export to csv for context / data check
    grouped_stats.to_csv('output/'+treatment_group+'_descriptive_stats.csv')
#Call function with appropriate variables
ols_regression(
    'experiment_one_groups', #add the treatment groups (experiment_one_groups,experiment_two_groups)
    '04/15/2024', #What is the date of the intervention?
    '03/15/2024', #What is the start date of the comparison timeframe?
    30) #How many days is the experiment and the comparison group going to last?
