# ----
# Author: Evgeny Chudaev
# Date: May, 2024
# Purpose: Code for some helper function for the Exploratory Data Analysis (EDA). 
# Updates: None
# ----

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pgeocode


def get_coordinates_from_zip(zip_code):
    """Obtain coordinates for a US zip code

    Args:
        zip_code (int): zip code

    Returns:
        tuple: latitude and longitude associated with the zip code
    """
    nomi = pgeocode.Nominatim('us')
    query = nomi.query_postal_code(zip_code)
    return query['latitude'], query['longitude']


def print_model_results(lr_model, predictors, outcome, housing_df):
    """Print models results for scikit learn leaner regression

    Args:
        lr_model (sklearn linear_model): Linear Regression model (sklearn only)
        predictors (list): list of predictor variables
        outcome (string): outcome variable name
        housing_df (Pandas DataFrame): DataFrame with housing data
    """
    
    print(f'Intercept: {lr_model.intercept_:.3f}')
    print('Coefficients:')
    for name,coeff in zip(predictors, lr_model.coef_):
        print(f'{name}: {coeff}')

    fitted = lr_model.predict(housing_df[predictors])
    RMSE = np.sqrt(mean_squared_error(housing_df[outcome], fitted))
    r2 = r2_score(housing_df[outcome], fitted)
    
    print(f'RMSE: {RMSE:.0f}')
    print(f'r2: {r2:.4f}')



def dummify_factors(df, predictors):
    """Turn categorical variable into dummy variable and drop first category to avoid multicollinearity (it becomes reference variable)

    Args:
        df (Pandas DataFrame): DataFrame containing predictor and other variables
        predictors (list): list of all predictor variables. However, only categorical ones will be dummified, others will be left as is.   

    Returns:
        Pandas DataFrame: DataFrame with predictor variables where categorical ones are dummified.
    """
    
    return pd.get_dummies(df[predictors], drop_first=True)   
