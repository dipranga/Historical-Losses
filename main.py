
import pandas as pd

###########################################################################

# 1. Data Consolidation
dataset = pd.read_csv('Historical_Losses_Data.csv')

# 1.1 Gathering Information about the dataset
dataset.shape
dataset.head(6)
dataset.tail(6)
dataset.info()

# 1.2 Removing NON-Significant Variables
dataset = dataset.drop(['Policy_Number'], axis=1)

# 1.3 Creating a list to gather variable/feature name
columns = dataset.columns.tolist()

###########################################################################

# 2. Exploratory Data Analysis -EDA Report
import seaborn as sns
import matplotlib.pyplot as plt
import pandasql as ps
plt.style.use('seaborn-whitegrid')   ### setting up Grid-Lines Style

##### 2.1 Univariate Analysis:  #####

# 2.1.1. finding ranges and catgories in variables
dataset.Age.sort_values().unique()
dataset.Years_of_Driving_Experience.sort_values().unique()
dataset.Number_of_Vehicles.sort_values().unique()
dataset.Gender.sort_values().unique()
dataset.Married.sort_values().unique()
dataset.Vehicle_Age.sort_values().unique()
dataset.Fuel_Type.sort_values().unique()
dataset.Losses.sort_values().unique()

# 2.1.2. Description:
Description = dataset.describe()
skew=dataset.skew()

# 2.1.3. Histogram of dataset
dataset.hist(bins=500, figsize=(20,15))
plt.savefig('Historical_Data(Histogram).png')
plt.show()

# taking a seperate look at 'Losses' variable because of high positive skewness:
dataset['Losses'].hist(bins=500, figsize=(10,7))
plt.title('Losses (Histogram)\n')
plt.savefig('Losses(Histogram).png')
plt.show()

# upper-capping 'losses' at 1200 according to industrial standards to deal with skewness;
Losses_Capped = dataset['Losses'].clip(upper=1200)
Losses_Capped.hist(bins=500, figsize=(10,7))
plt.title('Losses Capped (Histogram)\n')
plt.savefig('Losses_Capped(Histogram).png')
plt.show()
Losses_Capped.skew()

# making a single DataFrame of 'dataset' and 'Losses_Capped' for further analysis:
q1 = """ SELECT *,
        (CASE WHEN Losses > 1200 THEN 1200 ELSE Losses END) 
        AS Losses_Capped 
        FROM dataset """
Capped_Losses_dataset = ps.sqldf(q1, locals())

# 2.1.4. Pie-Plots:
# "Number_of_Vehicles" variable:
Number_of_Vehicles_Count = pd.DataFrame(
        dataset.Number_of_Vehicles.value_counts().reset_index())
Number_of_Vehicles_Count.columns=['Number of Vehicles','Count']
plt.pie(Number_of_Vehicles_Count['Count'],
        labels=Number_of_Vehicles_Count['Number of Vehicles'],
        autopct='%.1f%%', startangle=90)
plt.title('Number of Vehicles')
plt.savefig('Number_of_Vehicles_Count(Pie-Plot).png')
plt.show()

# taking a seperate view at charachter categorical data (Demographical Variables):

#Gender:
Gender_Count = pd.DataFrame(dataset.Gender.value_counts().reset_index())
Gender_Count.columns=['Gender','Count']
plt.pie(Gender_Count['Count'], labels=Gender_Count['Gender'], 
            autopct='%.1f%%', startangle=90)
plt.title('Gender Count')
plt.savefig('Gender_Count(Pie-Plot).png')
plt.show()

#Married:
Married_Count = pd.DataFrame(dataset.Married.value_counts().reset_index())
Married_Count.columns=['Married?','Count']
plt.pie(Married_Count['Count'], labels=Married_Count['Married?'],
            colors=['Yellow','Cyan'], autopct='%.1f%%', startangle=90)
plt.title('Married?')
plt.savefig('Married_Count(Pie-Plot).png')
plt.show()

#Fuel_Type:
Fuel_Type_Count=pd.DataFrame(dataset.Fuel_Type.value_counts().reset_index())
Fuel_Type_Count.columns=['Fuel_Type','Count']
plt.pie(Fuel_Type_Count['Count'], labels=Fuel_Type_Count['Fuel_Type'],
            colors=['y','Red'], autopct='%.1f%%', startangle=90)
plt.title('Fuel Type')
plt.savefig('Fuel_Type_Count(Pie-Plot).png')
plt.show()

# 2.1.5. Box-Plot of the variable for oulier detection:
dataset.boxplot(column=['Age','Years_of_Driving_Experience'], figsize=(5,10))
plt.savefig('Historical_dataset(Box-Plot)1.png', bbox_inches='tight')
plt.show()
dataset.boxplot(column=['Number_of_Vehicles','Vehicle_Age'], figsize=(5,10))
plt.savefig('Historical_dataset(Box-Plot)2.png', bbox_inches='tight')
plt.show()


##### 2.2 Bivariate Analysis: #####

# Dependent Variables are Losses & Losses_Capped 
# although we are performing analysis on Losses_Capped

#2.2.0. Correlation & Pair_Plots:
sns.heatmap(dataset.corr(), annot=True)
plt.title('Hiatorical Data Correlation (Heatmap)\n')
plt.savefig('Historical_Data_Correlation(Heatmap).png', bbox_inches='tight')
plt.show()

sns.pairplot(dataset, diag_kind='kde')
plt.savefig('Historical_Data(Pair-Plot).png')
plt.show()

# 2.2.1 Age vs Losses_Capped (Average) Analysing:
q1 = """ SELECT Age,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM Capped_Losses_dataset
        GROUP BY Age
        ORDER BY Age """

Age_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(Age_vs_Losses_Cap['Age'], Age_vs_Losses_Cap['Losses_Capped_Avg'], 'g.-')
plt.title('Age -vs- Losses (Capped)[Avg]\n')
plt.xlabel('Age')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Age-vs-Losses(Plot).png', bbox_inches='tight')
plt.show()

# 2.2.2 Years_of_Driving_Experience vs Losses_Capped(Avg) Analysing:
# YODE is used as an abbrivation of Years_of_Driving_Experience here.
q1= """ SELECT Years_of_Driving_Experience,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM Capped_Losses_dataset
        GROUP BY Years_of_Driving_Experience
        ORDER BY Years_of_Driving_Experience """

YODE_vs_Losses_Cap = ps.sqldf(q1, locals())

plt.plot(YODE_vs_Losses_Cap['Years_of_Driving_Experience'],
             YODE_vs_Losses_Cap['Losses_Capped_Avg'], 
             color='Brown', marker='.',linestyle='-')

plt.title('Years of Driving Experience -vs- Losses(Capped)[Avg]\n')
plt.xlabel('Years of Driving Experience')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Years_of_Driving_Experience-vs-Losses(Plot).png', bbox_inches='tight')
plt.show()

# 2.2.3  Number_of_Vehicles vs Losses_Capped(Avg) Analysing:
q1= """ SELECT Number_of_Vehicles,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM Capped_Losses_dataset
        GROUP BY Number_of_Vehicles
        ORDER BY NUmber_of_Vehicles """

Number_of_Vehicles_vs_Losses_Cap = ps.sqldf(q1, locals())

plt.plot(Number_of_Vehicles_vs_Losses_Cap['Number_of_Vehicles'],
             Number_of_Vehicles_vs_Losses_Cap['Losses_Capped_Avg'], 'bo-.')
plt.title('Number of Vehicles -vs- Losses (Capped)[Avg]\n')
plt.xlabel('Number of Vehicles')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Number_of_Vehicles-vs-Losses(Plot).png', bbox_inches='tight')
plt.show()

# 2.2.4 Gender vs Losses_Capped(Avg) Analysis:
q1= """ SELECT Gender,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM Capped_Losses_dataset
        GROUP BY Gender
        ORDER BY Gender """

Gender_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(Gender_vs_Losses_Cap['Gender'],
             Gender_vs_Losses_Cap['Losses_Capped_Avg'], 'co-.')
plt.title('Gender -vs- Losses (Capped)[Avg]\n')
plt.xlabel('Gender')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Gender-vs-Losses(Plot).png', bbox_inches='tight')
plt.show()

#2.2.5 Married vs Losses_Capped(Avg) Analysis:
q1= """ SELECT Married,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM Capped_Losses_dataset
        GROUP BY Married
        ORDER BY Married """

Married_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(Married_vs_Losses_Cap['Married'],
             Married_vs_Losses_Cap['Losses_Capped_Avg'], 'ro-.')
plt.title('Married? -vs- Losses (Capped)[Avg]\n')
plt.xlabel('Married?')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Married_vs_Losses(Plot).png', bbox_inches='tight')
plt.show()

# 2.2.6 Vehicle_Age vs Losses_Capped(Avg) Analysis:
q1= """ SELECT Vehicle_Age,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM Capped_Losses_dataset
        GROUP BY Vehicle_Age
        ORDER BY Vehicle_Age """

Vehicle_Age_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(Vehicle_Age_vs_Losses_Cap['Vehicle_Age'], 
             Vehicle_Age_vs_Losses_Cap['Losses_Capped_Avg'], 'mo-.')
plt.title('Vehicle Age -vs- Losses (capped)[Avg]\n')
plt.xlabel('Vehicle Age')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Vehicle_Age-vs-Losses(Plot).png', bbox_inches='tight')
plt.show()

# 2.2.7 Fuel_Type vs Losses_Capped(Avg) Analysis:
q1= """ SELECT Fuel_Type,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM Capped_Losses_dataset
        GROUP BY Fuel_Type
        ORDER BY Fuel_Type """

Fuel_Type_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(Fuel_Type_vs_Losses_Cap['Fuel_Type'],
             Fuel_Type_vs_Losses_Cap['Losses_Capped_Avg'], 'yo-.')
plt.title('Fuel Type -vs- Losses (Capped)[Avg]\n')
plt.xlabel('Fuel Type')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Fuel_Type-vs-Losses(Plot).png', bbox_inches='tight')
plt.show()

# EDA Report Ends.
##############################################

### 2.3 EDA Conclusion: ###

""" From the above EDA Report we get to know that 
    “Age”, “Years_of_Driving_Experience” and “Vehicle_Age” needs bucketing"""

# 2.4 Bucketing:
# 2.4.1. Age Bucketing:

q1 = """ SELECT *,
        (CASE 
             WHEN Age BETWEEN 16 AND 25 THEN 21 
             WHEN Age BETWEEN 26 AND 59 THEN 43 
             WHEN Age BETWEEN 60 AND 70 THEN 65 
             ELSE Age
             END) 
        AS Age_Bucket 
        FROM Capped_Losses_dataset 
        ORDER BY Age"""

CLD_Age_Bucket = ps.sqldf(q1, locals())

#2.4.2. Years_of_Driving_Experirncr Bucketing:

q1 = """ SELECT *,
        (CASE
             WHEN Years_of_Driving_Experience BETWEEN 0 AND 8 THEN 4
             WHEN Years_of_Driving_Experience BETWEEN 9 AND 40 THEN 25
             WHEN Years_of_Driving_Experience BETWEEN 41 and 53 THEN 47
             ELSE Years_of_Driving_Experience
             END)
        AS YODE_Bucket 
        FROM CLD_Age_Bucket
        ORDER BY Years_of_Driving_Experience"""

CLD_YODE_Bucket = ps.sqldf(q1, locals())

#2.4.3. Vehicle_Age Bucketing:

q1= """ SELECT *,
        (CASE 
             WHEN Vehicle_Age BETWEEN 0 AND 5 THEN 3
             WHEN Vehicle_Age BETWEEN 6 AND 10 THEN 8
             WHEN Vehicle_Age BETWEEN 11 AND 15 THEN 13
             ELSE Vehicle_Age
             END)
        AS V_Age_Bucket
        FROM CLD_YODE_Bucket
        ORDER BY Age"""

CLD_V_Age_Bucket = ps.sqldf(q1, locals())


# dropping variables that we have bucketed & capped
dataset_final = CLD_V_Age_Bucket.drop(columns=['Age',
                                    'Years_of_Driving_Experience',
                                    'Vehicle_Age','Losses'], axis=1)

###########################################################################

##### 3 Confirmatory Data Analysis: #####

# 3.1 gathering information about 'dataset_final':
Description_final = dataset_final.describe()
Skew_final = dataset_final.skew()

# 3.2 Bivariate Analysis (CDA):

# 3.2.1 Correlation Check:
sns.heatmap(dataset_final.corr(), annot=True)
plt.title('Final Dataset Corrlation\n')
plt.savefig('dataset_final_correlation(Heatmap).png', bbox_inches='tight')
plt.show()

# "Years_of_Driving_Experience" & "Age" still have high correlation even after bucketing
# there should be two models in model preperation because of this correlation

# 3.2.2 Age_Bucket vs Losses_Capped(Avg) :

q1 = """ SELECT Age_Bucket,
         AVG(Losses_Capped)
         AS Losses_Capped_Avg
         FROM dataset_final
         GROUP BY Age_Bucket
         ORDER BY Age_Bucket """

Age_Bucket_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(Age_Bucket_vs_Losses_Cap['Age_Bucket'],
             Age_Bucket_vs_Losses_Cap['Losses_Capped_Avg'], 'g.-')
plt.title('Age Bucket -vs- Losses (Capped)[Avg]\n')
plt.xlabel('Age (Bucket)')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('Age_Bucket-vs-Losses_Cap(Plot).png', bbox_inches='tight')
plt.show()

#3.2.3 YODE_Bucket vs Losses_Capped(Avg) :

q1 =""" SELECT YODE_Bucket,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM dataset_final
        GROUP BY YODE_bucket
        ORDER BY YODE_Bucket """

YODE_Bucket_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(YODE_Bucket_vs_Losses_Cap['YODE_Bucket'], 
             YODE_Bucket_vs_Losses_Cap['Losses_Capped_Avg'],
             color='Brown', marker='.',linestyle='-')

plt.title('YODE_Bucket -vs- Losses(Capped)[Avg]\n')
plt.xlabel('Years of Driving Experience (Bucket)')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('YODE_Bucket-vs-Losses_Cap(Plot).png', bbox_inches='tight')
plt.show()

# 3.2.4 V_Age_Bucket vs Losses_Capped(Avg):

q1 = """ SELECT V_Age_Bucket,
        AVG(Losses_Capped)
        AS Losses_Capped_Avg
        FROM dataset_final
        GROUP BY V_Age_Bucket
        ORDER BY V_Age_Bucket """

V_Age_Bucket_vs_Losses_Cap = ps.sqldf(q1, locals())
plt.plot(V_Age_Bucket_vs_Losses_Cap['V_Age_Bucket'], 
         V_Age_Bucket_vs_Losses_Cap['Losses_Capped_Avg'],
         'mo-.')
plt.title('Vehicle Age -vs- Losses (capped)[Avg]\n')
plt.xlabel('Vehicle Age (Bucket)')
plt.ylabel('Losses_Capped_Avg')
plt.savefig('V_Age_Bucket-vs-Losses_Cap(Plot).png', bbox_inches='tight')
plt.show()

###########################################################################

##### 4 Data Pre-Processing: ##### (on 'dataset_final')

dataset_final_OLS = dataset_final.copy() # making a copy 

# 4.1 Missing Value Treatment:
" There are no missing value so skipping this step"

# 4.2 Oulier Treatment :
" Outlier Values have been treated above with the help of 'Capping' and 'Bucketing' "

# 4.3 Variable Transformation :
from sklearn.preprocessing import LabelEncoder

# 4.3.1 converting categorical data into binary/numerical form:
myencoder = LabelEncoder()

dataset_final_OLS['Gender'] = myencoder.fit_transform(dataset_final_OLS['Gender'])
dataset_final_OLS['Married'] = myencoder.fit_transform(dataset_final_OLS['Married'])
dataset_final_OLS['Fuel_Type'] = myencoder.fit_transform(dataset_final_OLS['Fuel_Type'])

# 4.4 Variable Creation :

# Segregation between Independent and Dependent Vaariables:
# X = Independent Variables ; Y = Dependent Variables

X = pd.DataFrame(dataset_final_OLS.drop(columns='Losses_Capped')).copy()
Y = pd.DataFrame(dataset_final_OLS['Losses_Capped']).copy()

# from CDA we know "YODE_Bucket" & "Age_Bucket" have correlation nearly to 1
# making two seperate models for these variables
X_Age = pd.DataFrame(X.drop(columns='YODE_Bucket')).copy()
X_YODE = pd.DataFrame(X.drop(columns='Age_Bucket')).copy()

############################################################################

##### 5. Model Development:   ######
import statsmodels.formula.api as smfa
import statsmodels.tools as smt

# 5.1 Creating Linear Formula:
# b₀X₀ + b₁X₁ + b₂X₂ + ......... + b₈X₈
# where X₀ should be constant
X = smt.add_constant(X)

# 5.2 Creating Different Models with OLS (Ordinary Least Square) 
####   & using Backward Elimination Approach

import selectionprocess as sp   ## user generated module 
# setting a significance level of 5% (i.e, 0.05)
sig_level = 0.05

# 5.2.1 Model-1 : on X_Age
X_Age_Modeled, Age_Model_Summary = sp.BackwardElimination_OLS_DataFrame(X_Age, Y, sig_level)

# "Number of_Vehicles" is removed by backward elimination through OLS
# because its p-value > significance level

font_dict = {'family' : 'monospace',
             'size'   : 'large',
             'weight' : 'semibold'}

plt.text(0, 0, str(Age_Model_Summary),font_dict)
plt.axis('off')
plt.savefig('OLS_report_X_Age_Modeled.png', bbox_inches='tight')
plt.show()

# 5.2.2 Model-2 : on X_YODE
X_YODE_Modeled, YODE_Model_Summary = sp.BackwardElimination_OLS_DataFrame(X_YODE, Y, sig_level)

# "Number of_Vehicles" is removed by backward elimination through OLS
# because its p-value > significance level

plt.text(0, 0, str(YODE_Model_Summary),font_dict)
plt.axis('off')
plt.savefig('OLS_report_X_YODE_Modeled.png', bbox_inches='tight')
plt.show()


""" selecting Model-1 for further prediction 
            as it has higher R² value and lower AIC value """

coeff = smfa.OLS(Y, X_Age_Modeled).fit().params

# 5.2.3 Splitting into Train Test Values:
from sklearn.model_selection import train_test_split

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Age_Modeled, Y,
                                                test_size=0.2, random_state=66)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_Train, Y_Train)

###########################################################################################

##### 6. Model Performance:  #####

Y_Predicted = regressor.predict(X_Test)

# 6.1 Creating a single Analysis DataFrame:

Y_Predicted_temp = pd.DataFrame(Y_Predicted, columns=['Losses_Predicted'])
Y_Test_temp = pd.DataFrame(Y_Test.reset_index().drop('index',axis=1))

temp = dataset_final.copy()
tempTrain, tempTest=train_test_split(temp,test_size=0.2,random_state=66)

temp = tempTest.drop(columns=['Losses_Capped','YODE_Bucket'])
temp = pd.DataFrame(temp.reset_index().drop('index',axis=1))

Analysis = pd.concat([temp, Y_Test_temp, Y_Predicted_temp], axis=1)
Error_Values = pd.DataFrame(Analysis['Losses_Predicted']-Analysis['Losses_Capped'],
                            columns=['Error (Losses_Predicted-Losses_Capped)'])

Analysis = pd.concat([Analysis, Error_Values], axis=1)

#END.

