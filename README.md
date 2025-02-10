
# Attendance OLS Regression
**Context** - The following code demonstrates hypothesis testing for an intervention on attendance for school partners. The purpose of this code is to:

 1. Take the two files shared ("sample_attendance" and "sample_attendance_historical") and clean the data
 2. Group students into treatments based on whether they are in the control or the treatment group for each experiment
 3. Display the result of an OLS regression, boxplot and histogram to understand whether there is a significant change in the means between the treatment and control groups before and after an intervention date. 
	
**Requirements**:
1.  Ensure you have Python 3.7+ installed on your system.
    
2.  Clone or download this repository to your local machine.
    
3.  Open a terminal/command prompt and navigate to the project directory.
    
4.  Install the required packages:
    
    ```
    pip install pandas
    pip install matplotlib
    pip install statsmodels
    pip install numpy
    pip install seaborn

 4.  Adjust the following fields from lines 89-92 on run.py to reflect the (1)Treatment Groups (2) The Intervention Date (3) The start date of your comparison period and (4) The Time Window
