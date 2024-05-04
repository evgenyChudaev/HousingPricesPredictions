# ----
# Author: Evgeny Chudaev
# Date: May, 2024
# Purpose: Front-end Streamlit app for Exploratory Data Analysis (EDA)
# Updates: None
# ----

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sys
import pathlib
from joblib import load
import pickle
from statsmodels.regression.linear_model import OLSResults

# NEEDED FOR EMAIL LEAD SCORING TO BE DETECTED
# APPEND PROJECT DIRECTORY TO PYTHONPATH
# Streamlit does not recognize our Python path so have to make it absolute 
# then set new path for Strimlit where it search for files.\ and modules

working_dir = str(pathlib.Path().absolute())

print(working_dir)
sys.path.append(working_dir)

# Page config
st.set_page_config(page_title="Housing Price Prediction App", layout="wide")

st.title("Exploratory data analysis")

st.write("To select and engineer relevant features for the sales price prediction model, I've conducted an extensive exploratory data analysis (EDA) to identify the most important data attributes, interrelations between variables, and anomalies. This EDA phase encompassed fitting a Linear Regression model to illustrate the significance of various variables and their impact on the outcome variable. Additionally, the Linear Regression model acts as a reliable benchmark for the more advanced models employed in predicting property sales prices within the Price Prediction section.")

st.write("Below, you'll find some of the results from this EDA grouped by topic. Click on each section to view the details.")


#   CACHING DATA - NEEDED TO PREVENT REQUIRING DATA TO BE RE-INPUT
@st.cache_data()
def load_housing_datasets():
    """Load housing datasets created in the EDA pipeline step

    Returns:
       Pandas DataFrames: several dataframes used in visualizations and analysis in this module
    """
    
    
    # main housing dataset
    housing_df = pd.read_pickle('./00_data/housing_df_with_coords.pkl')    
    
    # dataset with 1 factor variable - property type
    df_housing_with_factor = pd.read_pickle('./00_data/housing_df_with_factor.pkl')
    
    # dataframe with zip group
    housing_df_with_zipgroup = pd.read_pickle('./00_data/housing_df_with_zip_groups.pkl')
    
    # group main housing dataset by zip code and coords
    housing_df_group = pd.read_pickle('./00_data/housing_df_group.pkl')
    
    
     # group main housing dataset by Year sold and property type
    housing_df_sales_by_year = pd.read_pickle('./00_data/housing_df_sales_by_year.pkl')
    
    # Corr matrix
    corr = pd.read_pickle('./00_data/feature_corr.pkl')
    
    # housing without outliers
    housing_df_no_outliers = pd.read_pickle('./00_data/housing_df_with_zipgroup_no_outliers.pkl')
    
    
    loaded_dataframes = \
    (
        housing_df, 
        df_housing_with_factor, 
        housing_df_with_zipgroup, 
        housing_df_group, 
        housing_df_sales_by_year,
        corr,
        housing_df_no_outliers        
    ) 
    
    return loaded_dataframes


@st.cache_data()
def load_ols_models():
    """Load several linear regression models to be used in this module

    Returns:
        Multiple: Linear regression models (fitted)
    """
    
    # simple OLS model without factor variables
    house_lm_simple = load('./00_data/house_lm_simple.joblib')
    
    # simple OLS model with 1 factor variable - property type
    house_lm_single_factor = \
         OLSResults.load('./00_data/housing_df_with_factor_statsmodel.pkl')
    
    # OLS model with interaction terms SqFtTotLiving*ZipGroup
    house_lm_interaction_terms = \
        OLSResults.load('./00_data/house_lm_statsmodel_with_interaction_terms.pkl')
    
    house_lm_interaction_terms_no_outliers = \
        OLSResults.load('./00_data/house_lm_statsmodel_with_interaction_terms_no_outliers.pkl')
    
    models = (
        house_lm_simple, 
        house_lm_single_factor, 
        house_lm_interaction_terms, 
        house_lm_interaction_terms_no_outliers
    )
    
    return models


@st.cache_data()
def display_map(df, color, size, hover_name, lat, lon):
    """Return map figure (plotly scatterplot)

    Args:
        df (Pandas DataFrame): DataFrame containing market details for each zip code
        color (float): continuous variable with color gradient for the map
        size (float): size of the buble on the map
        hover_name (various): additional details shown on the tooltip
        lat (float): latitude
        lon (float): longitude

    Returns:
        figure: plotly express figure
    """
    
    fig = px.scatter_mapbox(df, 
                            lat=lat, 
                            lon= lon, 
                            hover_name= hover_name ,
                            size= size,
                            color= color, 
                            color_continuous_scale=px.colors.cyclical.IceFire
                            )
    fig.update_layout(mapbox_style='open-street-map')
       
    return fig


# Load datasets
housing_df, \
df_housing_with_factor, \
housing_df_with_zipgroup, \
housing_df_group, \
housing_df_sales_by_year,  \
corr, \
housing_df_no_outliers \
    = load_housing_datasets()


# Load regression models
house_lm_simple, \
house_lm_single_factor, \
house_lm_interaction_terms, \
house_lm_interaction_terms_no_outliers \
    = load_ols_models()


# !!!Each step of the exploratory data analysis (EDA) is shown in one of the expander containers below.

# Map with zip codes and related information
with st.expander("GEOGRAPHY BY ZIP CODE"):    
    
    st.write('The dataset comprises 80 zip codes, with the two priciest ones situated in Mercer Island and Bellevue. In these areas, the median property prices were up to three times higher than those outside the city core. Conversely, less expensive zip codes are found outside the city core, where the majority of sales and purchase transactions occurred. This suggests that location and other geographical characteristics could serve as strong predictors of housing prices. The exploration of location features will be conducted later as part of the exploratory data analysis (EDA) process.')
        
    px_map = display_map(
        housing_df_group, 
        'AdjSalePrice-median',
        'Count-count',
        'AdjSalePrice-median',
        'latitude',
        'longitude'
        )
    st.plotly_chart(px_map)

# Outliers
with st.expander("OUTLIERS"):
           
    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["About Outliers", "Sale Price",
                                                "Property Type",                                
                                                "Year of Sale",
                                                "Bedrooms",
                                                "Bathrooms",
                                                "Living Space (sq ft)"])
    
    
    with tab0:
        st.write("Outliers, which are data points significantly different from others in a dataset, can profoundly impact predictive model performance. They can distort statistical measures, reduce model robustness, increase model complexity, and lead to inaccurate predictions, especially in linear models. ")
    
        st.write("Identifying and excluding outliers should be a crucial part of any exploratory data analysis (EDA) process. While there isn't a definitive method for identifying and removing outliers, one straightforward approach is using box plots to identify points lying outside the whiskers (based on the interquartile range, or IQR).")
        
        st.write("Despite the limitations and subjectivity of this heuristic approach, it suffices for this project's purposes. Other simple methods include z-score filtering and the percentile method. More sophisticated techniques involve multivariate methods (evaluating distance from the mean across multiple variables) and various machine learning methods like the Isolation Forest.")
    
    with tab1:
        
        st.write("There seem to be outlier properties with exceptionally high and low sales prices. In the upcoming tabs, we'll delve further into these outliers and contemplate excluding them later if it will enhance our model.")
        
        fig = px.box(housing_df,  y="AdjSalePrice")
        st.plotly_chart(fig)
        
    with tab2:        
        st.write("Most of the outliers are among the single family type property with one property approaching $12M in value.")        
        fig = px.box(housing_df, x="PropertyType", y="AdjSalePrice")
        st.plotly_chart(fig)

    
    with tab3:
        st.write("Our dataset comprises property sale transactions from 2006 to 2015. The peak median sales price in the county was observed in 2012 and declined for the following three years. It's important to note that the dataset used in this analysis may have been generated from synthetic data and may not accurately reflect the true market conditions at that time.")
        fig = px.box(housing_df_no_outliers , x="Year_Sold", y="AdjSalePrice")
        st.plotly_chart(fig)

    
    with tab4:
        st.write("There is a very distinct outlier with 30+ bedrooms.")
        fig = px.box(housing_df,  y='Bedrooms')
        st.plotly_chart(fig)

    with tab5:
        st.write("In the County, the average property typically features 2-3 bathrooms, with the maximum number reaching up to 8.")
        fig = px.box(housing_df,  y='Bathrooms')
        st.plotly_chart(fig)

    with tab6:
        st.write("Square footage ranges widely, spanning from as little as 370 square feet to over 10,000 square feet. Clear outliers exist, possibly due to sheds or other non-livable structures at the lower end of the scale, and exceptionally large, one-of-a-kind mega mansions at the higher end.")
        fig = px.box(housing_df,  y='SqFtTotLiving')
        st.plotly_chart(fig)



# Show histogram for sales price, sqft by property type
with st.expander("SALES PRICES"):
    
    st.write("Sales prices vary widely and represent the primary source of outliers. Single-family properties exhibit the broadest range of prices, spanning from nearly 0 to over 10 million USD. These outliers will be addressed at a later stage.")   
    fig = px.histogram(housing_df, x="AdjSalePrice",  color='PropertyType', marginal="rug",
                    title='Distribution of Sales Prices',
                    hover_data=housing_df.columns)
    st.plotly_chart(fig)

# Cumulative density plots
with st.expander("CUMULATIVE DENSITY"):  
    
    
    st.write("Cumulative density plots serve as valuable tools for visually representing the distribution of a dataset. They facilitate a clear understanding of the likelihood of observing values below or above specific thresholds, aiding in the detection of outliers.")  
    
    tab1, tab2, tab3 = st.tabs(["Adjusted Sales Price",                                
                                "Bedrooms",
                                "SqFtTotLiving"])
    
    with tab1: 
        st.write("The cumulative density plot for sales prices indicates that there is only a 0.3% probability of observing sales transactions of 3 million USD or higher. We will utilize $3 million USD as a threshold and subsequently exclude properties exceeding this price point from the model.") 
        fig = px.ecdf(housing_df, x="AdjSalePrice")
        st.plotly_chart(fig)    
    
    with tab2:        
        st.write("Similar to the sales price, the cumulative density plot for the number of bedrooms indicates that there is less than 0.3% probability of observing sales transactions for properties with 6 or more bedrooms. We will utilize 6 bedrooms as a threshold and subsequently exclude properties exceeding this number of bedrooms from the model.") 
        fig = px.ecdf(housing_df, x="Bedrooms")
        st.plotly_chart(fig)
    
    with tab3:
        st.write("The cumulative density plot for the sqaure footage of living space indicates that there is less than 0.3% probability of observing sales transactions for properties with 6200 or more square feet of living space. We will utilize 6200 sq ft of living space as a threshold and subsequently exclude properties exceeding this threshold from the model.") 
        fig = px.ecdf(housing_df, x="SqFtTotLiving")
        st.plotly_chart(fig)   


# Correlation matrix
#We use this matrix to help us make a decision on the relevant features.
# Does not include categorical variables.
# Use correlation funnel (Pytimetk package) in the future that accounts for categorical variables
with st.expander("FEATURE CORRELATIONS"):    
    
    tab1, tab2,= st.tabs(["About correlation matrix",                                
                            "The matrix"
                            ])
    with tab1:
        st.write("The purpose of a feature correlation matrix is to quantify and visualize the relationships between different features (variables) in a dataset. The correlation matrix provides insights into how closely related or dependent one feature is to another. This information helps in identifying redundant or highly correlated features, which can be important for feature selection and dimensionality reduction in machine learning models, as well as for understanding the underlying structure of the data. Additionally, the correlation matrix aids in identifying potential multicollinearity issues that may affect the interpretability and stability of regression models.") 
        
        st.write("The current feature matrix comprises unadjusted variables in their original form, prior to any feature engineering processes, aimed at conducting an initial assessment of correlations among the variables and identifying promising candidates for inclusion in the model. This matrix excludes categorical and other engineered variables, which will be introduced later in the analysis.")
    
        st.write("A more comprehensive feature correlation analysis can be conducted using the Python package Pytimetk and its correlation funnel method. Exploration of this package will be deferred to a later stage.")
        
    with tab2:
        st.write("The correlation matrix shows that the sales price has a medium/high high correlation with the following variables: SqFtTotLiving, Bedrooms, Bathrooms, BldgGrade. These features are strong candidates for inclusion in the model. Additional features will also be engineered later.")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, cmap='Blues', annot=True)
        plt.title('Correlation Matrix')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Display the heatmap in Streamlit
        st.pyplot()
        
# Analysis of residuals
with st.expander("RESIDUAL ANALYSIS"): 
    
    tab1, tab2, tab3 = st.tabs(["About residuals",                                
                            "Heteroskedasticity",
                            "Partial Residuals"
                            ])
    
    with tab1:
        st.write('Exploratory data analysis often involves constructing a Linear Regression model (or other linear models) and assessing its diagnostics to gain insights into drivers, influential variables, and the potential impact of changes in model coefficients on the outcome variable being predicted. This process aids in developing intuition, which can then inform the construction of more advanced models and the interpretation of results.')
        
        st.write("When conducting diagnostics for Linear Regression, it's crucial to examine the distribution of residuals. This analysis is pertinent not only for formal hypothesis testing but also for identifying model limitations or incompleteness, particularly if residuals deviate from a normal distribution.")
    
    with tab2:
        st.write("Heteroskedasticity refers to situations where the variance of the residuals is unequal over a range of measured values.")
        
        st.write("Understanding of variance of residuals is important for the validity of the Linear Regression model. While other more complex models will be used for making predictions in this project, it is nevertheless important to understand whether Linear Regression is valid for the purspose of the interpretation of coefficients which is part of the Exploratory Data Analysis.")
        
        st.write("As shown in the plot below, the variance of residuals appears to increase for properties with lower and higher values. Although this observation may violate one of the distribution assumptions, it won't impact the predictive ability of the non-linear models employed to forecast sales prices in this project. Nevertheless, we will interpret the coefficients of the Linear Regression cautiously and will also exclude higher-valued homes above $3 million USD, as determined by the outlier analysis conducted earlier.")        
        
        figx = plt.imread('./00_data/partial_residual_plot1.png') 
        fig, ax = plt.subplots() 
        ax.axis('off') 
        ax.imshow(figx) 
        st.pyplot()
    
        
    with tab3:    
        st.write("Visualizing partial residuals aids in explaining the relationship between predictor and outcome variables in the Linear Regression Model. A partial residual plot isolates the relationship between a single predictor variable and the outcome variable, while considering all other predictor variables. It illustrates the estimated contribution that the predictor variable (in this case, SqFtTotLiving) makes to the sales price.")  
        
        st.write("As evident from the below plot, the regression line underestimates the sales price for properties with larger square footage. Knowing this may be helpful in drawing conclusions about larger properties or for excluding certain property sizes from the analysis.")       
        
        
        figx = plt.imread('./00_data/partial_residual_plot.png') 
        fig, ax = plt.subplots() 
        ax.axis('off') 
        ax.imshow(figx) 
        st.pyplot()

# Trends
with st.expander("SALES TRENDS"):       
    
    st.write("As seen in the plots below, the median sales price for all property types in the County decreased from its peak. However, it remains uncertain whether this decline reflects the true economic environment or stems from limitations within the dataset. Additionally, it's important to refrain from drawing inferences from any models in this project beyond the date ranges covered by the dataset.") 
    
    
    tab1, tab2 = st.tabs(["Sales Price",                                
                                "Volume"])   
    
    
    with tab1:
        fig = px.line(housing_df_sales_by_year, x='Year_Sold', y='AdjSalePrice', color='PropertyType', symbol="PropertyType")
        st.plotly_chart(fig)
    
    with tab2:
        fig = px.line(housing_df_sales_by_year, x='Year_Sold', y='Count', color='PropertyType', symbol="PropertyType")
        st.plotly_chart(fig)
   

# Pairplots that describe potential relationships between 2 pairs of variables
with st.expander("OTHER FEATURE RELATIONSHIPS"): 
    tab1, tab2 = st.tabs(["Before removing outliers",                                
                                "After removing outliers"])
    
    with tab1:
        
        st.write("This pairplot represents the relationship between pairs of variables BEFORE OUTLIERS ARE REMOVED in the dataset and helps identify possible relationships between variables, and outliers in the data. It is clear that there are outliers in the data in terms of number of bedrooms, sales price and square footage. As noted earlier, these some of these outliers can affect robustness of our model. The diagonal of the plot shows the relationship of the variable with itself.")  
           
        fig = px.scatter_matrix(housing_df,
            dimensions=["AdjSalePrice", "SqFtLot", "SqFtTotLiving", "YrBuilt", "Bedrooms", "Bathrooms", 'BldgGrade'],
            color="PropertyType")
        st.plotly_chart(fig)
    
    with tab2:
        
        st.write("After removing outliers, the interrelationships between certain variables become clearer. For instance, there is a discernible correlation between square footage and sales price, which is further supported by the correlation matrix. It appears that these outcome and predictor variables are moving in the same direction. Similar observations can be made regarding sales price and the number of bathrooms and building grade, although the relationship with the square footage of the lot may not be as robust. Nonetheless, comparing variables on the pair plot before and after outlier removal provides an additional tool for understanding interrelationships, alongside the feature correlation matrix.")
        
        fig = px.scatter_matrix(housing_df_no_outliers,
            dimensions=["AdjSalePrice", "SqFtLot", "SqFtTotLiving", "YrBuilt", "Bedrooms", "Bathrooms", 'BldgGrade'],
            color="PropertyType")
        st.plotly_chart(fig)


# Details of each zip group
with st.expander("GEOSPACIAL FEATURE ENGINEERING"):
       
    tab0, tab1, tab2 = st.tabs(["About geospacial features", "Zip Group Details",    
                         "Zip Group Visuals"])    
    
    with tab0:
        st.write("In the realm of real estate, location often stands out as one of the most influential predictors of sales price. While this dataset does not provide coordinates, Zip Code can serve as a proxy. However, a significant challenge arises when using Zip Code as-is due to considerable variability in sales volume within each zip code, ranging from as low as 1 to as high as 788. Additionally, employing a categorical variable with 80 distinct values would result in 80 different coefficients, which would not aid in model explainability. Therefore, it is advisable to group Zip codes in a manner that preserves the informativeness of this feature while simultaneously reducing the number of distinct values.")
    
        st.write("One method for grouping Zip Codes involves utilizing another variable such as median residual. Alternatively, other approaches may involve grouping by median sale price. In this instance, we will create five groups of Zip Codes, each containing approximately the same number of sales transactions. Subsequently, the Zip group will be incorporated as a new categorical feature in the model, and we will evaluate the model's performance before and after introducing this variable.")
        
    with tab1:  
              
        col1, col2, col3 = st.columns(3)
        
        with col1:        
            st.metric(label="Total Zip Codes", value=len(housing_df_group)) 
            st.metric(label="Sales Transactions", value=housing_df_group['Count-count'].sum())         
              
          
        with col2: 
            st.metric(label="Max # of sales by ZIP", value=housing_df_group['Count-count'].max())            
            st.metric(label="Min # of sales by ZIP", value=housing_df_group['Count-count'].min())            
        
        with col3:
            max_zip_ind = housing_df_group['Count-count'].idxmax()
            min_zip_ind = housing_df_group['Count-count'].idxmin()            
            st.metric(label="ZIP with most sales", value=housing_df_group.iloc[max_zip_ind][0])            
            st.metric(label="ZIP with least sales", value=housing_df_group.iloc[min_zip_ind][0])                    
        
        st.dataframe(housing_df_group[['ZipCode', 'Count-count', 'AdjSalePrice-median', 'ZipGroup_str']])        
       
    
    with tab2:      
        st.write("Zip codes are organized into buckets of roughly equal size based on the median residual. The map below also marks the location of each zip group. Interestingly, the Zip Groups seem to cluster together on the map.")
        fig = px.bar(housing_df_group, x='ZipGroup_str', y='Count-count' )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)
        
        px_map = display_map(
        housing_df_group, 
        'ZipGroup_str',
        'Count-count',
        'AdjSalePrice-median',
        'latitude',
        'longitude'
        )
        
        st.plotly_chart(px_map)

# Linear regression analysis - 3 models: before any feature engineering, after some feature engineering, and after geature engineering + outlier removal
with st.expander("LINEAR REGRESSION ANALYSIS"):   
    
    
    tab0, tab1, tab2, tab3 = st.tabs(["About Linear Regression", "No feature engineering", 
                                "After engineering zip groups", 
                                "After excluding outliers"])    
    
    with tab0:
        st.write("This section includes the results from 3 iterations of the Linear Regression model:")
        st.write("1. Before any feature engineering (includes only key variables without any changes).")
        st.write("2. After introducing a new categorical variable with Zip groups and adding interaction term based on square footage and zip group.")
        st.write("3. After excluding outliers based on price and square footage.")
        
        st.write("As shown by the results, the performance of the Linear Regression model significantly improved after each iteration, particularly with the introduction of the Zip group and interaction term. After all, in the real estate world, location is crucial; however, the combination of location (i.e., zip group) and square footage also played a significant role. Additionally, excluding outliers further enhanced the model.")
        
        st.write("Carefully reviewing the interaction terms based on the zip group and square footage, we can conclude that the adding square footage in the most experive location dispropotionately increases the predicted sales price bya  factor of almost 3 compared with the increased from adding a square foot on average.")
        
        st.write("The Linear Regression Model, with its inherent explainability, will also serve as a benchmark for the more complex models used in the 'Price Prediction' section of this web application.")    
    
    with tab1:
        st.subheader("Linear Regression - results before zip code grouping")    
        st.write("This model explains approximately 54% of variance of the outcome variable as coefficient of determination shows. Square footage of the lot appears not to be statistically significant variable in the model as the p-value for the coefficient indicates.")
        st.write("By examining the values of other coefficients, we can deduce that adding an extra foot of living space increases the selling price by $233. Similarly, increasing the building grade by one level boosts the selling price by 109,400 USD, assuming all other coefficients remain constant. However, initially, it may seem counterintuitive that adding the number of bathrooms and bedrooms decreases the selling price.")
        st.write("The explanation lies in the high correlation between the Bedrooms, Bathrooms, and SquareFtTotLiving variables (as indicated in the correlation matrix shown earlier). This correlation suggests that adding additional bedrooms or bathrooms without increasing the square footage of the house will lower the selling price. After all, who would want to purchase a house with more (albeit smaller) rooms without an increase in living space? Therefore, correlated variables can indeed diminish the explainability of the model.")
                
        st.write(house_lm_single_factor.summary())
    
    with tab2:
        st.subheader("Linear Regression - results after introducing zip code grouping and interaction term, outliers present")        
        st.write("As mentioned previously, to incorporate a location proxy, we include the Zip group. Location could serve as a confounding variable, and its exclusion might compromise the model's robustness. After all, location stands as one of the primary factors influencing selling prices in the real estate industry. Further details on how the Zip group was derived can be found in the section titled 'Geospatial Feature Engineering' above.")
        st.write("Another addition to this model is the interaction between two variables: Zip group and the square footage of living space. It is reasonable to assume that the size of the house impacts the selling price differently across various locations. For instance, increasing living space in a high-end location may elevate the value of the house more than adding the same amount of living space in another area.")
        st.write("For instance, adding an extra square foot in the last zip group could increase the selling price of the house by 342 USD (consisting of 114 USD for SqFtTotLiving and 228 USD for SqFtTotLiving.ZipGroup[T.4]). This amount is nearly three times higher than the increase seen when adding a square foot in the first zip code area (114 USD for SqFtTotLiving and 0 USD for Zip Group 1, which serves as a reference variable for other zip groups).")
        
        st.warning("Due to the correlation between variables, it is important to interpret coefficients and their directions with great caution.")
        st.write(house_lm_interaction_terms.summary())
        
    with tab3:
        st.subheader("Linear Regression - results after introducing zip code grouping, outliers excluded")
        st.write("Excluding outliers in the selling price and number of bedrooms, as discussed earlier, further improves the performance of the model. This iteration of the Linear Regression model now explains more than 72% of the variance in the selling price.")
        st.warning("Due to the correlation between variables, it is important to interpret coefficients and their directions with great caution.")        
        st.write(house_lm_interaction_terms_no_outliers.summary())

# Wrap up and next steps
with st.expander("CONCLUSION AND NEXT STEPS"):   
    st.write("This thorough exploratory data analysis (EDA) has enhanced understanding of the importance of various predictor variables. Additionally, certain obvious outliers were excluded, and confounding variables based on location were introduced, resulting in a significant improvement in model performance. However, given the correlation between variables, it is essential to interpret coefficients and their direction with caution.")
    
    st.write("Next, I'll construct several other non-linear models using the PyCaret package in Python and will offer users a straightforward and intuitive interface for predicting the value of houses in King County, Washington State.")
    
