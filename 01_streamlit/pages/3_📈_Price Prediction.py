# ----
# Author: Evgeny Chudaev
# Date: May, 2024
# Purpose: Front-end Streamlit app for Price prediction
# Updates: None
# ----


import streamlit as st
import pandas as pd
from pycaret.regression import *

@st.cache_data()
def load_top_models():
    """Load 3 best models to make property price predictions 

    Returns:
        pickled models: 3 regression models used for prediction
    """
    best_model_0 = load_model('./04_models/best_model_00')
    best_model_1 = load_model('./04_models/best_model_01')
    best_model_2 = load_model('./04_models/best_model_02') 
    
    return best_model_0, best_model_1, best_model_2


@st.cache_data()
def load_zipcode_ref_datasets():     
    """Load 2 pickled dataframes used in the drop down menus on the form and one dataframe with 
       additional information about location (zip code)

    Returns:
        Pandas DataFrames & Pandas Series: several Pandas DataFrames and Pandas Series
    """
    
    # dataframe with zip group
    housing_df_with_zipgroup = pd.read_pickle('./00_data/housing_df_with_zipgroup_no_outliers.pkl')    
    #housing_df_with_zipgroup.info()    
       
    housing_df_zipgroup_ref = pd.read_pickle('./00_data/housing_df_group.pkl')
    housing_df_zipgroup_ref['ZipGroup'] =  housing_df_zipgroup_ref['ZipGroup_str'].astype('category') 
    #housing_df_zipgroup_ref.info()    

    housing_df_propType_ref = housing_df_with_zipgroup['PropertyType'].unique()
    buildingGrade_ref = housing_df_with_zipgroup['BldgGrade']
    bathroomCount_ref = housing_df_with_zipgroup['Bathrooms']
    bedroomCount_ref = housing_df_with_zipgroup['Bedrooms']    
    sqftLiving_ref = housing_df_with_zipgroup['SqFtTotLiving']   
    sqftLot_ref = housing_df_with_zipgroup['SqFtLot']   
    
           
    return (housing_df_zipgroup_ref, 
            housing_df_propType_ref, 
            buildingGrade_ref, 
            bathroomCount_ref, 
            bedroomCount_ref,
            sqftLiving_ref,
            sqftLot_ref)
    

# load necessary datasets
zip_code_dataset, \
housing_df_propType_ref, \
buildingGrade_ref, \
bathroomCount_ref, \
bedroomCount_ref, \
sqftLiving_ref, \
sqftLot_ref = load_zipcode_ref_datasets()

# load top 3 regression models
best_model1, best_model2, best_model3 = load_top_models()


st.write("Please choose a zip code, adjust any other relevant parameters, and then click the ‘GET PRICE PREDICTIONS’ button to obtain property value estimates using three different models.")

# input form 
with st.form("input_form"):   
    
    row1 = st.columns([0.3, 0.7])
    zipCode =   row1[0].selectbox('Select Zip Code', zip_code_dataset['ZipCode'].tolist())
    propType = row1[1].selectbox('Select Property Type', housing_df_propType_ref)
    
    zipCodeDetails = zip_code_dataset[zip_code_dataset['ZipCode'] == zipCode]  
        
    row2 = st.columns([0.5,0.5])
    sqftLiving= row2[0].slider('Sq Ft (living space)', 300, int(sqftLiving_ref.max()), int(sqftLiving_ref.mean()), step=10)
    sqftLot= row2[1].slider('Sq Ft (lot size)', 100, int(sqftLot_ref.max()), int(sqftLot_ref.mean()), step=10)
        
    row3 = st.columns([0.3,0.4,0.3])
    bedrooms  = row3[0].slider('Number of bedrooms',  int(bedroomCount_ref.min()), int(bedroomCount_ref.max()), int(bedroomCount_ref.mean()), step=1)
    bathrooms = row3[1].slider('Number of bathrooms',  int(bathroomCount_ref.min()), int(bathroomCount_ref.max()), int(bathroomCount_ref.mean()), step=1)
    bldggrade = row3[2].slider('Building grade', int(buildingGrade_ref.min()), int(buildingGrade_ref.max()), int(buildingGrade_ref.mean()), step=1) 
    
    select_dict = {
        'zipCode': [zipCode],
        'PropertyType': [propType],
        'SqFtTotLiving': [sqftLiving],
        'SqFtLot': [sqftLot],
        'Bedrooms': [bedrooms],
        'Bathrooms': [bathrooms],
        'BldgGrade': [bldggrade],
        'ZipGroup' : [int(zipCodeDetails['ZipGroup'])]
    }
    
    prediction_df = pd.DataFrame(select_dict)
    
    st.markdown("""---""") 
    
    if st.form_submit_button('--> GET PRICE PREDICTIONS <--'):        
        
        st.subheader("Property value predictions with 3 models")        
        
        col1, col2, col3 = st.columns(3)
        
        # prediction for property value using 3 models
        with col1: 
            text = f"$ {int(predict_model(best_model1, prediction_df)['prediction_label']):,}"            
            st.metric(label="Gradient Boosting Regressor", value= text)
        
        with col2:
             text = f"$ {int(predict_model(best_model2, prediction_df)['prediction_label']):,}" 
             st.metric(label="LGBM Regressor", value=text)
        
        with col3:
             text = f"$ {int(predict_model(best_model3, prediction_df)['prediction_label']):,}" 
             st.metric(label="CatBoostRegressor", value=text)
         
        
        st.markdown("""---""") 
        col1, col2 = st.columns(2)
        
        with col1:
            st.map(zipCodeDetails[['latitude', 'longitude']])
        
        with col2:
            st.subheader(f"Details for zip code {zipCode}")
            
            text = f"${int(zipCodeDetails['AdjSalePrice-median']):,}"
            st.metric(label="Median sales price", value= text)
            
            text = f"{int(zipCodeDetails['SqFtTotLiving-median']):,}"
            st.metric(label="Median living area (sq ft)", value= text) 
                        
            text = f"{int(zipCodeDetails['SqFtLot-median']):,}"
            st.metric(label="Median lot size (sq ft)", value= text)      

            text = f"{int(zipCodeDetails['YrBuilt-median'])}"
            st.metric(label="Median year of property construction", value= text)
