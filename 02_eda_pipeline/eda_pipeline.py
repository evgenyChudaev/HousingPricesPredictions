# ----
# Author: Evgeny Chudaev
# Date: May, 2024
# Purpose: Code for Exploratory Data Analysis (EDA). Results of the EDA are saved as .pkl (datagrames and models) or .png (figures)
# Updates: None
# ----

import pandas as pd
from sklearn import linear_model as lm
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf 
from statsmodels.stats.outliers_influence import OLSInfluence
from joblib import dump, load
import pickle
from statsmodels.regression.linear_model import OLSResults
import matplotlib.pyplot as plt
import seaborn as sns

from support_functions import (get_coordinates_from_zip, 
                              print_model_results, 
                              dummify_factors)

# Forsome models, figures and processes it takes a long time to fit/plot/process,
# therefore it makes sense not to fit/plot them every time unless it's necessary
FIT_MODELS = False
ASSIGN_COORDS = False
LOAD_DF_FROM_FILE = False
PLOT_FIGURES = False 

# Load housing dataset from a CSV file
if LOAD_DF_FROM_FILE:
    housing_df = pd.read_csv("./00_data/house_sales.csv", sep='\t')
    housing_df.head()
    

# assign coordinates and year soldbased on zip code 
if ASSIGN_COORDS:
    housing_df[['latitude', 'longitude']]  = \
        housing_df.apply(lambda row: get_coordinates_from_zip(row['ZipCode']),                              
                                    result_type='expand',
                                    axis=1
                                    )
    housing_df['DocumentDate'] = pd.to_datetime(housing_df['DocumentDate'])
    housing_df['Year_Sold'] = housing_df['DocumentDate'].dt.year     
    housing_df.to_pickle('./00_data/housing_df_with_coords.pkl')

    
# read housing data with coordinates into dataframe from a pickled file
housing_df = pd.read_pickle('./00_data/housing_df_with_coords.pkl')

# remove NAs
housing_df[housing_df.isna().any(axis=1)]
housing_df.info()

# 1. --LINEAR REGRESSIONS WITH NUMERIC VARIABLES ONLY
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms',
          'Bedrooms', 'BldgGrade']

outcome = 'AdjSalePrice'

if FIT_MODELS:    
    house_lm = lm.LinearRegression()
    house_lm.fit(housing_df[predictors], housing_df[outcome])

    dump(house_lm, './00_data/house_lm_simple.joblib')


# load fitted model model
house_lm_simple = load('./00_data/house_lm_simple.joblib')
# print model results
print_model_results(house_lm_simple, predictors, outcome, housing_df)


# --2 ADDITIONAL ANALYSIS - outliers, influential variables, feature correlations

# 2.1 Correlation plot
# feature correlation matrix
corr = housing_df[['AdjSalePrice', 'SqFtLot', 'SqFtTotLiving','Bedrooms', 'Bathrooms', 'BldgGrade', 'YrBuilt', 'Year_Sold']].corr()
corr.to_pickle('./00_data/feature_corr.pkl')

plt.figure(figsize=(8, 6))
sns.heatmap(corr, cmap='Blues', annot=True)
plt.title('Correlation Matrix')
plt.show()


# 2.2 More outliers by zip code
if FIT_MODELS:   
    predictors = ['SqFtTotLiving',
        'SqFtLot',
        'Bathrooms',
        'Bedrooms',
        'BldgGrade']
    outcome = 'AdjSalePrice'

    housing_df['AdjSalePrice'].mean()

    house_outlier = sm.OLS(
        housing_df[outcome],
        housing_df[predictors].assign(const=1)
        )

    result = house_outlier.fit()

# 2.3 Analysis of residuals
# OBSERVATIONS:
# Heteroskedasticity - variance of residuals tend to increase 
# Distribution of residuals is relevant for the validity of formal statistic a inference.
# Normally distributed errors are a sign that model is complete.
# Here we observe a lack of constant residual variance acrsoss the range of predicted values.
# The variance of residuals tends to increase for higher priced homes but is also large for lower-valued
# properties. 
if PLOT_FIGURES:  
    fig, ax = plt.subplots(figsize=(5,5))
    sns_plot = sns.regplot(
        x = result.fittedvalues, 
        y =np.abs(result.resid),
        scatter_kws={'alpha': 0.25}, 
        line_kws={'color': 'C1'},
        lowess=True, 
        ax=ax
    )

    ax.set_xlabel('predicted')
    ax.set_ylabel('abs(residual)')    
    fig.savefig('./00_data/partial_residual_plot1.png') 
    

figx = plt.imread('./00_data/partial_residual_plot1.png') 
fig, ax = plt.subplots() 
ax.axis('off') 
ax.imshow(figx) 

 
 
# 2.3 Analysis of influential variables
if PLOT_FIGURES:  
    influence = OLSInfluence(result)
    sresiduals = influence.resid_studentized_internal
    sresiduals.idxmin(), sresiduals.min()

    influence = OLSInfluence(result)
    fig, ax = plt.subplots(figsize=(5,5))
    ax.axhline(-2.5, linestyle='--', color='C1')
    ax.axhline(2.5, linestyle='--', color = 'C1')
    ax.scatter(influence.hat_matrix_diag, 
            influence.resid_studentized_internal, 
            s=1000 * np.sqrt(influence.cooks_distance[0]), 
            alpha=0.5)

    ax.set_xlabel('hat values')
    ax.set_ylabel('studentized residuals')
    
    pickle.dump(fig, open('./00_data/influence_plot.pkl', 'wb'))

figx = pickle.load(open('./00_data/influence_plot.pkl', 'rb'))
figx.show() 

# 2.4 Analysis of partial residuals
# partial residual plot - care should be taken about inferences for larger properties with sq ft > 3000
if PLOT_FIGURES:  
    fig = sm.graphics.plot_ccpr(result, 'SqFtTotLiving')
    # can't save above figure as pkl, saving as png instead
    fig.savefig('./00_data/partial_residual_plot.png') 
    

figx = plt.imread('./00_data/partial_residual_plot.png') 
fig, ax = plt.subplots() 
ax.axis('off') 
ax.imshow(figx) 


# 3. --LINEAR REGRESSIONS WITH PROPERTY TYPE CATEGORICAL VARIABLE DUMMIFIED - slight improvement in model

# 3.1 Fit with Scikit-learn
predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 
              'Bedrooms', 'BldgGrade', 'PropertyType']

outcome = 'AdjSalePrice'

X = dummify_factors(housing_df , predictors)
Y = housing_df[outcome]


if FIT_MODELS:  
    house_lm_factor = lm.LinearRegression()
    house_lm_factor.fit(X, Y)

    dump(house_lm_factor, './00_data/house_lm_factor.joblib')
    # SAVE X & Y
    pd.concat([X, Y], axis=1).to_pickle('./00_data/housing_df_with_factor.pkl') 

 # load model
house_lm_factor = load('./00_data/house_lm_factor.joblib')
# load dataframe with factor
df_housing_with_factor = pd.read_pickle('./00_data/housing_df_with_factor.pkl')

print_model_results(house_lm_factor, 
                    [col for col in list(df_housing_with_factor.columns) if col !=outcome ], 
                    outcome, 
                    df_housing_with_factor 
                    )


# 3.2 Fit with statsmodel instead as its more informative.
if FIT_MODELS: 
    
    housing_df['PropertyType'] = housing_df['PropertyType'].astype('category') 
    model = smf.ols(formula='AdjSalePrice ~ SqFtTotLiving +  \
        SqFtLot + Bathrooms + Bedrooms + BldgGrade + PropertyType', data=housing_df)
    results = model.fit()
    
    results.save('./00_data/housing_df_with_factor_statsmodel.pkl')

# load model
house_lm_factor = OLSResults.load('./00_data/housing_df_with_factor_statsmodel.pkl')
house_lm_factor.summary()


# 4. --LINEAR REGRESSIONS WITH PROPERTY TYPE CATEGORICAL VARIABLE + ZIPGROUP - group zip codes. Significant improvement of model

# 4.1. exploring factor variable zipcode - there are 80 zipcodes, they should be grouped to several buckets to be used in the model
pd.DataFrame(housing_df['ZipCode'].value_counts())

pd.DataFrame(housing_df['ZipCode'].value_counts()).max()[0]
pd.DataFrame(housing_df['ZipCode'].value_counts()).min()[0]


# 4.2. Create groups of zip codes based on residual
# Zip code is an important feature that may be a proxy of location of the property.
# Including all 80 zip codes may not be practical as many have only a small number of sales transactions
# One approach is to group zip codes by the median residual, other approaches may also include grouping by median sale price.

if FIT_MODELS:  
    
    housing_df_mod = housing_df.copy()
    housing_df_mod['residual'] = housing_df_mod[outcome] - house_lm_factor.predict(X)
    hdf = housing_df_mod.groupby(['ZipCode']).agg(
                                            {
                                            'ZipCode': 'size',
                                            'residual': np.median   
                                            }
                                        ) \
                                        .sort_values('residual')
                                        
    hdf.columns = ['count', 'median_residual']

    hdf['cum_count'] = np.cumsum(hdf['count'])
    hdf['ZipGroup'] = pd.qcut(hdf['cum_count'], 5, labels=False, retbins=False)

    to_join = hdf[['ZipGroup']]  
    housing_df = housing_df.join(to_join, on='ZipCode')
    housing_df['ZipGroup'] = housing_df['ZipGroup'].astype('category')

    housing_df.to_pickle('./00_data/housing_df_with_zip_groups.pkl')

                                 
    
    # load dataframe
    housing_df = pd.read_pickle('./00_data/housing_df_with_zip_groups.pkl')

    # 4.3. Fit a new model with interaction terms - superimportant and increases model r-squared and other metrics significantly.
    # after adding zip group we get a significant improvement in R-squared and RMSE   

    model = smf.ols(formula='AdjSalePrice ~ SqFtTotLiving*ZipGroup +  \
        SqFtLot + Bathrooms + Bedrooms+ BldgGrade + PropertyType', data=housing_df)
    results = model.fit()
    
    results.save('./00_data/house_lm_statsmodel_with_interaction_terms.pkl')

# load model
new_results = OLSResults.load('./00_data/house_lm_statsmodel_with_interaction_terms.pkl')
new_results.summary()


# --5. GROUP MAIN DATAFRAME FOR USE IN VISUALS

# group main housing dataset by zip code and coords
if FIT_MODELS:  
    housing_df['DocumentDate'] = pd.to_datetime(housing_df['DocumentDate'])
    housing_df['Year_Sold'] = housing_df['DocumentDate'].dt.year   
    housing_df['ZipGroup_str'] = housing_df['ZipGroup'].astype(str)
    
    housing_df_group = housing_df.groupby(['ZipCode', 'ZipGroup_str', 'latitude', 'longitude']). \
        agg(
            {
                'AdjSalePrice': [np.median, np.max, np.mean],
                'SqFtLot' : np.median,
                'SqFtTotLiving' : np.median,
                'Bathrooms': np.median,
                'Bedrooms': np.median,
                'YrBuilt': np.median,            
                'PropertyID': 'count'
            }
        ).reset_index() \
        .rename(columns={'PropertyID': 'Count'})
        
    housing_df_group.columns = \
        [("-").join(col) if col[1] !='' else col[0] for col in housing_df_group.columns]   
    housing_df_group['ZipCode'] = housing_df_group['ZipCode'].astype(str)
    housing_df_group.to_pickle('./00_data/housing_df_group.pkl')

    # group main housing dataset by Year sold and property type
    housing_df_sales_by_year= housing_df.groupby(['Year_Sold','PropertyType']). \
        agg(
            {
                'AdjSalePrice': np.median,                                  
                'PropertyID': 'count'
            }
        ).reset_index() \
        .rename(columns={'PropertyID': 'Count'})

    housing_df_sales_by_year.to_pickle('./00_data/housing_df_sales_by_year.pkl')


housing_df_group = pd.read_pickle('./00_data/housing_df_group.pkl')
housing_df_sales_by_year = pd.read_pickle('./00_data/housing_df_sales_by_year.pkl')

housing_df_group.head()
housing_df_sales_by_year.head() 



# 6. --REMOVE OUTLIERS USING HEURISTIC APPROACH. LATER USE MORE SOPHISTICATED METHODS SUCH AS ISOLATION FOREST ETC.
# There is no established method to separate outliers from non-outliers, we will use a hearistic approach 
# and will explore other methods such as isolation forest etc. in the future.

housing_df_no_outliers = housing_df[
                                    (housing_df['Bedrooms'] <= 6) & 
                                    (housing_df['AdjSalePrice'] <= 3000000)                                    
                                    ]

housing_df_no_outliers.to_pickle('./00_data/housing_df_with_zipgroup_no_outliers.pkl')

# 6.1 - fit model without outliers
if FIT_MODELS:  
    model = smf.ols(formula='AdjSalePrice ~ SqFtTotLiving*ZipGroup +  \
        SqFtLot + Bathrooms + Bedrooms+ BldgGrade + PropertyType', data=housing_df_no_outliers)
    results = model.fit()
    
    results.save('./00_data/house_lm_statsmodel_with_interaction_terms_no_outliers.pkl')


# load model
results_no_outliers = OLSResults.load('./00_data/house_lm_statsmodel_with_interaction_terms_no_outliers.pkl')
results_no_outliers.summary()


