# ----
# Author: Evgeny Chudaev
# Date: May, 2024
# Purpose: Front-end Streamlit app for Price prediction
# Updates: None
# ----

import os
import pandas as pd
from pycaret.regression import *


# loading housing data
housing_df_no_outliers = pd.read_pickle('./00_data/housing_df_with_zipgroup_no_outliers.pkl')
housing_df_no_outliers.info()

# keep only relevant features and a label
housing_df_no_outliers = housing_df_no_outliers[[
    'ZipGroup', 'PropertyType', 'SqFtTotLiving', 'SqFtLot',
    'Bedrooms', 'Bathrooms', 'BldgGrade', 'AdjSalePrice' 
]]


regression = setup(data = housing_df_no_outliers , 
                    target = 'AdjSalePrice', 
                    categorical_features = ['PropertyType', 'ZipGroup'],
                    session_id=123,
                    # K-fold
                    fold_strategy='stratifiedkfold',
                    fold = 5,                    
                    train_size = 0.8,
                    preprocess=True,                    
                    verbose=False              
                    )
 
models = compare_models()

# Running All Available Models and selecting best 3
best_models = compare_models(
    sort = 'RMSE',
    n_select= 3
)


get_metrics()

# Get the grid
pull()

# Top 3 Models
best_models[0]
best_models[1]
best_models[2]

# tune models - does not always improve default model
tuned_best_0 = tune_model(best_models[0])
tuned_best_1 = tune_model(best_models[1])
tuned_best_2 = tune_model(best_models[2])

# predict on test data (20%)
predict_model(best_models[0]) 
predict_model(best_models[1]) 
predict_model(best_models[2]) 


# Finalize 3 best models. Best practice is to finalize. PyCaret retrains on the whole dataset
best_model_0_finalized = finalize_model(best_models[0])
best_model_1_finalized = finalize_model(best_models[1])
best_model_2_finalized = finalize_model(best_models[2])


# Save / load model
os.mkdir('./models')

# Save top 3 models
save_model(
    model = best_model_0_finalized,
    model_name = './04_models/best_model_00'
)

save_model(
    model = best_model_1_finalized,
    model_name = './04_models/best_model_01'
)

save_model(
    model = best_model_2_finalized,
    model_name = './04_models/best_model_02'
)

best_model_0 = load_model('./04_models/best_model_00')
best_model_1 = load_model('./04_models/best_model_01')
best_model_2 = load_model('./04_models/best_model_02')

# predict unseen data with loaded model - test that model works
df = pd.DataFrame(
    {'ZipGroup': [1],
    'PropertyType': ['Multiplex'],
    'SqFtTotLiving': [2400],
    'SqFtLot': [9373],
    'Bedrooms': [6],
    'Bathrooms': [3.00],
    'BldgGrade': [7]}
)

predict_model(best_model_2, df)

# Summary Plot: Overall top features
interpret_model(best_models[2], 
                    plot='summary', 
                    use_train_data=True)








