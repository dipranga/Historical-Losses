# Slection Process all in one module

import numpy as np
import pandas as pd
import statsmodels.formula.api as smfa

# 1 Backward Elimination: for array-like
# independent_array == array-like, independent dataset/variables (normally denoted by X in program)
# dependent_array == array-like(1d), dependent dataset/variable
# significance level == float, if value of significance is 5% set this value as 0.05

# returns(arrray-like) ==  moduled or processed final array after removing all 
#                           in-significant variables through OLS
# returns (summary) == summary of final model as 'statsmodels.iolib.summary.Summary'
# also prints final summary

def Backward_Elimination_OLS_Array(independent_array, dependent_array, significance):
    count = len(independent_array[0])
    for num1 in range(count):
        OLS_Regressor = smfa.OLS(dependent_array, independent_array).fit()
        Max_P_value = max(OLS_Regressor.pvalues).astype(float)
        
        if Max_P_value > significance:
            for num2 in range(count-num1):
                if(OLS_Regressor.pvalues[num2].astype(float) == Max_P_value):
                    independent_array = np.delete(arr=independent_array, 
                                                  obj=num2, axis=1)
    print(OLS_Regressor.summary())
    return independent_array, OLS_Regressor.summary()


# 2 Backward Elimination: for DataFrame-like
# independent_DF == DataFrame-like, independent dataset/variables (normally denoted by X in program)
# dependent_DF == DataFrame-like(1d), dependent dataset/variable
# significance level == float, if value of significance is 5% set this value as 0.05

# returns (DataFrame-like)==   moduled or processed final DataFrame after removing all 
#                               in-significant variables through OLS
# returns (summary) == summary of final model as 'statsmodels.iolib.summary.Summary'

# also prints final summary


def BackwardElimination_OLS_DataFrame(independent_DF, dependent_DF, significance):
    count = len(independent_DF.columns)
    for num1 in range(count):
        OLS_Regressor = smfa.OLS(dependent_DF, independent_DF).fit()
        Max_P_value = max(OLS_Regressor.pvalues)
        
        if Max_P_value > significance:
            for num2 in range(count-num1):
                if(OLS_Regressor.pvalues[num2] == Max_P_value):
                    independent_DF = pd.DataFrame.drop(independent_DF,
                                      columns=independent_DF.columns[num2],
                                      axis=1)
    print(OLS_Regressor.summary())
    return independent_DF, OLS_Regressor.summary()