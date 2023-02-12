# This python script contains all user-defined functions used for the data cleaning python notebook

def check_vif(df):
    """
    Function which checks the variance inflation factor between selected variables of interest
    
    Parameters:
    -----------
    df: pandas Dataframe
        dataset to be checked
        
    Returns:
    --------
    pandas Dataframe with 2 columns
        variables - name of columns in df
        vif - the variance inflation factor of the variables
                
    vif == 1: There is no correlation between a given predictor variable and any other predictor variables in the model.
    1 < vif <=5: There is moderate correlation between a given predictor variable and other predictor variables in the model.
    vif > 5: There is severe correlation between a given predictor variable and other predictor variables in the model.
    
    """
    
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_df = pd.DataFrame(df.columns, columns=['variables'])
    vif_df['vif'] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    
    vif_df.sort_values(by='vif', inplace=True)
    
    return vif_df



def get_outliers_idx(df):
    """
    Function which returns the index of the dataframe df which are outliers as defined as more than 1.5x interquartile range away from mean
    
    Parameters:
    -----------
    df: pandas Dataframe
        input dataframe of size (n by 1) to identify outliers
        
    Returns:
    --------
    pandas Dataframe of size (n by 1) of boolean values where True indicates an outlier
    
    """

    import pandas as pd
    
    quartiles = df.describe([.25, .75])[['25%','75%']]
    
    iqr = quartiles['75%'] - quartiles['25%']
    lower_bound = quartiles['25%'] - 1.5 * iqr
    upper_bound = quartiles['75%'] + 1.5 * iqr

    return ((df < lower_bound) | (df > upper_bound))


def generate_segments(X, outcome_var, var_in, var_out):
    """
    Function which separates dataset by var_in based on statistically distinct outcome_var, and the segments are labelled in var_out
    
    Parameters:
    -----------
    X: pandas dataframe
        dataset to be segmented
        
    outcome_var: string
        variable that will be tested for statistically different groups
        
    var_in: string
        variable that will be segmented
        
    var_out: string
        new variable name for the segments
        
    Returns:
    --------
    pandas dataframe
        Same dataframe as X with one additional column named var_out
        
    """
    from scipy.stats import mannwhitneyu
    
    # get the unique values first
    groups = X[var_in].unique()
    
    # create a var_out, if it exists, set all to NaN
    X[var_out] = None
    
    # start off with segment 1
    current_segment = 1
    # set the segment of the first group to 1
    X.loc[X[var_in] == groups[0], var_out] = current_segment
    
    for i, group1 in enumerate(groups):
        # check if the current group already has a segment allocation
        if X[X[var_in] == group1][var_out].isnull().sum():
            # if it does not, increment segment number by 1
            current_segment += 1
            # and set this group as the new segment
            X.loc[X[var_in] == group1, var_out] = current_segment
        else:
            # if it has already been assigned a segment, skip this iteration of the for loop
            continue
        
        # now go through all the remaining groups and compare it with this group
        for j, group2 in enumerate(groups[i+1:]):
            group1_data = X[X[var_in] == group1][outcome_var]
            group2_data = X[X[var_in] == group2][outcome_var]
            stat, p = mannwhitneyu(group1_data, group2_data)
            
            # check if statistically different i.e. p value less than 0.05
            if p > 0.05:
                # if not stat diff, assign to current segment
                X.loc[X[var_in] == group2, var_out] = current_segment
                
    # assign last group to last segment if there are still null
    X.loc[X[var_in].isnull(), var_out] = current_segment + 1
    
    # convert to integer
    X.loc[:, var_out].astype(int)
    
    # return X
    return X