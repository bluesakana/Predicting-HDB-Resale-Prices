from sklearn.pipeline import Pipeline

class SubsegmentModel(Pipeline):
    """
    Class to construct, fit and predict with a sub-segment model, i.e. dataset is split into groups and each group has its own fitted model. Note: model specifications will be the same for all groups, only the fitted model is different for each group.
    """
    
    def __init__(self, model, segment_var):
        """
        Method to instantiate object
        
        Parameters:
        -----------
        self: object
            The object itself
            
        model: sklearn object
            Some form of model or pipeline that has the fit and predict methods
            
        segment_var: string
            Name of variable that the data should be segmented by
            Ensure that the dataset X has this variable inside
        
        Returns:
        --------
        Null
        
        """
        self.model_segment = model # assign model_segment as the model selected (can be a pipeline)
        self.segment_var = segment_var
        self.model = {} # create empty dictionary of models
    
    def fit(self, X, y):
        """
        Method to fit the sub-segment models
        
        Parameters:
        -----------
        self: object
            The object itself
            
        X: pandas dataframe
            Predictor variables used for fitting, and ensure segment_var is one of the variables
            
        y: pandas series
            Outcome variable to be predicted
        
        Returns:
        --------
        Null
        
        """
        from copy import deepcopy
        
        # find the list of segments from the data
        segments = X[self.segment_var].unique()
        # for each segment, generate and fit model segment with data
        for segment in segments:
            # create a copy of the model segment
            self.model[segment] = deepcopy(self.model_segment)
        
            # create segments of X and y
            X_segment = X[X[self.segment_var] == segment]
            y_segment = y[X_segment.index]
            
            # fit model segment
            self.model[segment].fit(X_segment, y_segment)
            
        
    def predict(self, X):
        """
        Method to predict with the sub-segment models
        
        Parameters:
        -----------
        self: object
            The object itself
            
        X: pandas dataframe
            Predictor variables used for predicting, and ensure segment_var is one of the variables
        
        Returns:
        --------
        y_pred: pandas series
            Predicted outcome variables
        
        """
        import pandas as pd
        
        # create list of predictions
        y_pred = pd.Series(index=X.index, dtype='float')
        
        # find the list of segments from the data
        segments = X[self.segment_var].unique()        
        
        # run predict for each segment model on data
        for segment in segments:
            # get segment data
            X_segment = X[X[self.segment_var] == segment]
            
            # predict
            y_pred[X[self.segment_var] == segment] = self.model[segment].predict(X_segment)
        
        # return predictions
        return y_pred
    
    def score(self, X, y, sample_weight=None):
        """
        Function which returns the coefficient of determination R^2 of the prediction.
        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always
        predicts the expected value of y, disregarding the input features,
        would get a R^2 score of 0.0.
        
        Parameters:
        -----------
        X: pandas dataframe
            Predictor variables used for fitting, and ensure segment_var is one of the variables
            
        y: pandas series
            True outcome variable
            
        sample_weight: array-like
            Sample weights
            
        Returns:
        --------
        score: float
            R2 score of the model
            
        """
        from sklearn.metrics import r2_score
        
        y_pred = self.predict(X)
        
        #u = ((y - y_pred) ** 2).sum()
        #v = ((y - y.mean()) ** 2).sum()
        
        return r2_score(y, y_pred, sample_weight=sample_weight)
    
#     def get_params(self, deep=True):
#         """
#         Function which gets parameters for this estimator
        
#         Parameters
#         ----------
#         deep : bool, default=True
#             If True, will return the parameters for this estimator and
#             contained subobjects that are estimators.
        
#         Returns
#         -------
#         params : dict
#             Parameter names mapped to their values.
        
#         """
#         out = dict()
#         for key in self._get_param_names():
#             value = getattr(self, key)
#             if deep and hasattr(value, "get_params") and not isinstance(value, type):
#                 deep_items = value.get_params().items()
#                 out.update((key + "__" + k, val) for k, val in deep_items)
#             out[key] = value
#         return 
    
#     def _get_param_names(cls):
#         """Get parameter names for the estimator"""
#         # fetch the constructor or the original constructor before
#         # deprecation wrapping if any
#         init = getattr(cls.__init__, "deprecated_original", cls.__init__)
#         if init is object.__init__:
#             # No explicit constructor to introspect
#             return []

#         # introspect the constructor arguments to find the model parameters
#         # to represent
#         init_signature = inspect.signature(init)
#         # Consider the constructor parameters excluding 'self'
#         parameters = [
#             p
#             for p in init_signature.parameters.values()
#             if p.name != "self" and p.kind != p.VAR_KEYWORD
#         ]
#         for p in parameters:
#             if p.kind == p.VAR_POSITIONAL:
#                 raise RuntimeError(
#                     "scikit-learn estimators should always "
#                     "specify their parameters in the signature"
#                     " of their __init__ (no varargs)."
#                     " %s with constructor %s doesn't "
#                     " follow this convention." % (cls, init_signature)
#                 )
#         # Extract and sort argument names excluding 'self'
#         return sorted([p.name for p in parameters])