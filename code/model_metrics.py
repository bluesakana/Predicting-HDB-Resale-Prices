def test_metric(model, X_train, X_test, y_train, y_test, metric='r2_score', cv=5):
    """
    Function to return specified test metric for train, validation and test sets
    
    Parameters
    ----------
       
    model: sklearn model on the outcome variable
        model to run predictions on
    
    X_train: pandas Dataframe
        training predictor dataset
    
    X_test: pandas Dataframe
        test predictor dataset
        
    y_train: pandas Series/Dataframe
        training outcome dataset
        
    y_test: pandas Series/Dataframe
        test outcome dataset

    metric: str
        'r2_score' - evaluates R2 score (default)
        'rmse' - evaluates root mean squared error
        'rmape' - evaluates root mean absolute percentage error
        
    cv: int
        number of cross validation folds (default = 5)
        
    Returns
    -------
    Prints out metric in 3 lines
    
    """
    
    # import libraries
    from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # if modifier == 'linear':
    #     y_pred_train = model.predict(X_train)
    #     y_pred_test = model.predict(X_test)
    # elif modifier == 'log':
    #     y_pred_train = np.exp(model.predict(X_train))
    #     y_pred_test = np.exp(model.predict(X_test))
    #     y_train = np.exp(y_train)
    #     y_test = np.exp(y_test)
    
    # generate predictions
    #y_pred_train = model.predict(preproc.transform(X_train))
    #y_pred_test = model.predict(preproc.transform(X_test))
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # select function to use
    if metric == 'r2_score':
        train_metric = r2_score(y_train, y_pred_train)
        try:
            cross_metric = cross_val_score(model, X_train, y_train, scoring='r2')
        except: # except clause to catch custom class which does not have 'clone' methods needed for cross_val_score function to work. We write a custom cross_val_score and use this custom function instead
            cross_metric = cross_val_score_custom(model, X_train, y_train, cv=cv, scoring='r2')
        test_metric = r2_score(y_test, y_pred_test)
        
    elif metric == 'rmse':
        train_metric = mean_squared_error(y_train, y_pred_train, squared=False)
        try:
            cross_metric = -1 * cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
        except:
            cross_metric = -1 * cross_val_score_custom(model, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
        test_metric = mean_squared_error(y_test, y_pred_test, squared=False)
        
    elif metric == 'rmape':
        train_metric = mean_absolute_percentage_error(y_train, y_pred_train)
        try:
            cross_metric = -1 * cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_percentage_error')
        except:
            cross_metric = -1 * cross_val_score_custom(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_percentage_error')
        test_metric = mean_absolute_percentage_error(y_test, y_pred_test)
        
    else:
        print('ERROR: Invalid metric chosen')
        return
        
    print(f'Train {metric.upper()}:           \t{train_metric:.4f}')
    print(f'{cv}-Fold CV {metric.upper()}:     \t{np.nanmean(cross_metric):.4f}')
    print(f'Test {metric.upper()}:            \t{test_metric:.4f}')
    

def cross_val_score_custom(model, X, y, cv=5, scoring='r2'):
    """
    Function which computes the cross-validation score for custom models
    
    Parameters:
    -----------
    
    model: sklearn model
        The machine learning model to be evaluated
        
    X: pandas Dataframe
        Predictor dataset
        
    y: pandas Series/Dataframe
        Outcome dataset
        
    cv: int (default: 5)
        Number of folds to use in cross validation
            
    Returns:
    --------
    
    cv_scores: list
        List of cross-validation scores
            
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
    
    cv_scores = []
    
    # split data into number of cv folds and run predicts then save score for each
    for i in range(cv):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/cv)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        
        # compute score based on selected score type
        if scoring == 'r2':
            score = r2_score(y_val, y_pred)
        elif scoring == 'neg_root_mean_squared_error':
            score = mean_squared_error(y_val, y_pred, squared=False)
        elif scoring == 'neg_mean_absolute_percentage_error':
            score = mean_absolute_percentage_error(y_val, y_pred)
        else:
            print('ERROR: Invalid metric chosen')
            return
        
        cv_scores.append(score)
        
        return cv_scores