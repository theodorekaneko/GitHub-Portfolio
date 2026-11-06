import os
import kagglehub
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import d2_tweedie_score
from sklearn.model_selection import train_test_split
from dataframe import DataFrame
import streamlit as st
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor




def predict_loss_cost(VehPower, VehAge, DrivAge, BonusMalus, VehBrand, VehGas, Area, Density, Region, Exposure):

    #Define numerical and categorical columns
    numerical_columns = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    categorical_columns = ['VehBrand', 'VehGas', 'Area', 'Region']

    #Establish the path and route to both csv files
    path = kagglehub.dataset_download("karansarpal/fremtpl2-french-motor-tpl-insurance-claims")

    csv_array = os.listdir(path)

    csv_path_freq = os.path.join(path, 'freMTPL2freq.csv')
    csv_path_sev = os.path.join(path, 'freMTPL2sev.csv')

    df_freq = pd.read_csv(csv_path_freq)
    df_sev = pd.read_csv(csv_path_sev)

    #Pass each datatframe through the class: DataFrame
    df_freq = DataFrame(df_freq)
    df_sev = DataFrame(df_sev)

    #Filter both DataFrames
    sev_ids = set(df_sev.df['IDpol'].unique())
    freq_ids = set(df_freq.df['IDpol'].unique())
    common_ids = list(sev_ids.intersection(freq_ids))
    
    filtered_df_freq = df_freq.df[df_freq.df['IDpol'].isin(common_ids)]
    filtered_df_sev = df_sev.df[df_sev.df['IDpol'].isin(common_ids)]

    df_freq = DataFrame(filtered_df_freq)
    df_sev = DataFrame(filtered_df_sev)


    #Calculate pure premium and create a 1D array
    sorted_claim_amount = df_sev.claim_amount_by_ID('IDpol', 'ClaimAmount')
    sorted_exposure = df_sev.exposure_by_ID(df_freq)
    y = sorted_claim_amount / sorted_exposure

    y_train = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    

    #Create a 2D array of independent variables
    X_train = df_freq.get_2D_array(['IDpol', 'ClaimNb', 'Exposure'])

    #Get sample_weight/exposure
    exposure = df_freq.df['Exposure']

    #Scale and encode X_train
    preprocessor = ColumnTransformer([
        ('numerical', StandardScaler(), numerical_columns),
        ('categorical', OneHotEncoder(), categorical_columns)
    ])

    #Define the pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor', LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=1.5,
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        ))
    ])

    #Set a cap to avoid bias from extreme values
    loss_cap = np.percentile(y_train, 99)
    y_train_capped = np.minimum(y_train, loss_cap)

    #Fit the model
    model_pipeline.fit(X_train, y_train_capped, regressor__sample_weight=exposure)

    #Take user inputs
    user_inputs = {
        'VehPower': VehPower,
        'VehAge': VehAge,
        'DrivAge': DrivAge,
        'BonusMalus': BonusMalus,
        'VehBrand': VehBrand,
        'VehGas': VehGas,
        'Area': Area,
        'Density': Density,
        'Region': Region,
        'Exposure': Exposure
    }

    #Create Pandas DataFrame and define independent variables and exposure
    user_df = pd.DataFrame([user_inputs])
    user_X = user_df.drop(columns=['Exposure'])
    user_exposure = user_df['Exposure']

    #Calculate/predict amount per 1 unit of exposure
    predicted_rate = model_pipeline.predict(user_X)[0]

    #Calculate/predict final amount
    final_predicted_rate = predicted_rate * user_exposure.iloc[0]
    return final_predicted_rate

def test_regressor():

    numerical_columns = ['VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
    categorical_columns = ['VehBrand', 'VehGas', 'Area', 'Region']
        
    path = kagglehub.dataset_download("karansarpal/fremtpl2-french-motor-tpl-insurance-claims")
    
    csv_array = os.listdir(path)
    
    csv_path_freq = os.path.join(path, 'freMTPL2freq.csv')
    csv_path_sev = os.path.join(path, 'freMTPL2sev.csv')
    
    df_freq = pd.read_csv(csv_path_freq)
    df_sev = pd.read_csv(csv_path_sev)
    
    df_freq = DataFrame(df_freq)
    df_sev = DataFrame(df_sev)
    
    sev_ids = set(df_sev.df['IDpol'].unique())
    freq_ids = set(df_freq.df['IDpol'].unique())
    common_ids = list(sev_ids.intersection(freq_ids))
        
    filtered_df_freq = df_freq.df[df_freq.df['IDpol'].isin(common_ids)]
    filtered_df_sev = df_sev.df[df_sev.df['IDpol'].isin(common_ids)]
    
    df_freq = DataFrame(filtered_df_freq)
    df_sev = DataFrame(filtered_df_sev)

    claim_costs = df_sev.claim_amount_by_ID('IDpol', 'ClaimAmount')
    exposure_units = df_sev.exposure_by_ID(df_freq)
    y = claim_costs / exposure_units
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    

    X = df_freq.get_2D_array(['IDpol', 'ClaimNb', 'Exposure'])

        
    exposure = df_freq.df['Exposure']

    X_train, X_test, y_train, y_test, exposure_train, exposure_test = train_test_split(X, y, exposure, test_size=0.2, random_state=42)


    preprocessor = ColumnTransformer([
        ('numerical', StandardScaler(), numerical_columns),
        ('categorical', OneHotEncoder(), categorical_columns)
    ])
        
    
    model_pipeline = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor', LGBMRegressor(
            objective='tweedie',
            tweedie_variance_power=1.5,
            n_estimators=100,
            learning_rate=0.05,
            random_state=42
        ))
    ])

    loss_cap = np.percentile(y_train, 99)
    y_train_capped = np.minimum(y_train, loss_cap)

    
    model_pipeline.fit(X_train, y_train_capped, regressor__sample_weight=exposure_train)
    y_pred = model_pipeline.predict(X_test)

    d2 = d2_tweedie_score(y_test, y_pred, sample_weight=exposure_test, power=1.5)


    def decile_chart(y_test, y_pred, exposure_test):
        test_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred, 'Exposure': exposure_test})
        test_df['Decile'] = pd.qcut(y_pred, q=10, labels=False, duplicates='drop')

        def weighted_loss_cost(group):
            return group['Actual'].sum() / group['Exposure'].sum()

        weighted_deciles = test_df.groupby('Decile').apply(weighted_loss_cost)

        return weighted_deciles

    def check_negative(y_pred):
        num_negative = (y_pred < 0).sum()
        assert num_negative == 0, "Model predicted a negative cost"
        return num_negative

    lift = decile_chart(y_test, y_pred, exposure_test)
    num_negative = check_negative(y_pred)

    print(f"Tweedie Regressor D^2 Score: {d2:.4f}")
    print(f"\nActual loss costs by model-predicted risk decile: {lift}")
    print(f"\nNumber of negative predictions: {num_negative}")

if __name__ == '__main__':
    test_regressor()