# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
from joblib import load
import shap

#____________________________GENERAL FUNCTIONS_________________________________


@st.cache(allow_output_mutation=True)
def load_test_data():
    """cached function that returns our working examples for the dashboard"""
    return pd.read_csv("../../data/output_data/X_test.csv")

@st.cache(allow_output_mutation=True)
def load_training_data() : 
    """cached function thats returns the dataframe the model was trained on"""
    return pd.read_csv("../../data/output_data/X_train.csv")

@st.cache(allow_output_mutation=True)
def load_variable_description() : 
    '''cached function that returns variable description'''
    return pd.read_csv("../../data/output_data/new_variables_filled.csv")


#____________________________PREDICT FUNCTIONS_________________________________

@st.cache(allow_output_mutation=True)
def load_model() :
    """returns the estimator"""
    return load("../../model/estimator.joblib")

@st.cache(allow_output_mutation=True)
def compute_tree_explainer(model, X_train) : 
    """compute and returns the tree explainer based on a model and the training data"""
    data = X_train.drop(columns=["SK_ID_CURR"]).copy()
    #print("id data", hex(id(data)))
    explainer = shap.TreeExplainer(model.booster_, data=data.sample(2000, random_state=12),
                                   feature_dependence="interventional", output="probabilities")
    #print("id du explainer", hex(id(explainer)))
    return explainer
    

import requests
import json
import numpy as np

def predict_api(data) :
    url = 'https://ocp7-api.herokuapp.com'
    print(data.shape)
    j_data = data.to_json()
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    r = requests.post(url, data=j_data, headers=headers)
    
    return round(float(r.text.split(",")[1].replace("]]", ""))*100, 3)


#____________________________EXPLORER FUNCTIONS________________________________
    
@st.cache(allow_output_mutation=True)
def load_explorer_data():
    """cached function that load and returns all necessary data for exploration"""
    bb = pd.read_csv("../../data/output_data/test_bb_agg.csv")#, nrows=100000)
    bureau = pd.read_csv("../../data/output_data/test_bureau.csv", index_col=0)#, nrows=100000)
    ins = pd.read_csv("../../data/output_data/test_ins.csv", index_col=0)#, nrows=100000)
    ccb = pd.read_csv("../../data/output_data/test_ccb.csv")#, nrows=100000)
    pos = pd.read_csv("../../data/output_data/test_pos.csv")#, nrows=100000)
    prev = pd.read_csv("../../data/output_data/test_prev.csv", index_col=0)#, nrows=100000)
    return bb, bureau, ins, ccb, pos, prev

@st.cache(allow_output_mutation=True)
def load_raw_test_data(): 
    """load test data as exported from data source"""
    return pd.read_csv("../../data/input_data/application_test.csv")

#____________________________STATS FUNCTIONS___________________________________
@st.cache()
def load_raw_train_data(): 
    """load train data as exported from data source"""
    return pd.read_csv("../../data/input_data/application_train.csv")


