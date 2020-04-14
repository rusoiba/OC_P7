#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:58:30 2020

@author: Alex
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from bureau_explorer import bureau_explorer
from prev_explorer import prev_explorer
from application_explorer import application_explorer

@st.cache(allow_output_mutation=True)
def load_explorer_data():
    """cached function that load and returns all necessary data for exploration"""
    bb = pd.read_csv("../data/output_data/test_bb_agg.csv")#, nrows=100000)
    bureau = pd.read_csv("../data/output_data/test_bureau.csv", index_col=0)#, nrows=100000)
    ins = pd.read_csv("../data/output_data/test_ins.csv", index_col=0)#, nrows=100000)
    ccb = pd.read_csv("../data/output_data/test_ccb.csv")#, nrows=100000)
    pos = pd.read_csv("../data/output_data/test_pos.csv")#, nrows=100000)
    prev = pd.read_csv("../data/output_data/test_prev.csv", index_col=0)#, nrows=100000)
    return bb, bureau, ins, ccb, pos, prev

def load_variable_description() : 
    """cached function that returns variable description"""
    return pd.read_csv("../data/input_data/HomeCredit_columns_description.csv", index_col=0)
    
def explorer(id_curr) : 
    """function that returns the explorer page enabling further filtering"""
    bb, bureau, ins, ccb, pos, prev = load_explorer_data()
    variables = load_variable_description()
    
    source = st.selectbox("Source of information", ["Current Application", "Credit Bureau", "Home Credit"])
    
    
    if source == "Current Application" :
        application_explorer(id_curr)
        
    elif source == "Credit Bureau" :
        #Select the relevant information relative to the parameter passed to explorer
        ids_bureau = bureau.loc[bureau["SK_ID_CURR"]==id_curr,"SK_ID_BUREAU"]
        subset_bureau = bureau.loc[bureau["SK_ID_CURR"]==id_curr, :].set_index("SK_ID_BUREAU")
        subset_bb = bb.loc[bb["SK_ID_BUREAU"].isin(ids_bureau), :]
        
        bureau_explorer(subset_bureau, subset_bb, variables)
        
    elif source == "Home Credit" :
        ids_prev = prev.loc[prev["SK_ID_CURR"]==id_curr, "SK_ID_PREV"]
        
        subset_prev = prev.loc[prev["SK_ID_CURR"]==id_curr, :].set_index("SK_ID_PREV").drop("SK_ID_CURR", axis=1)
        subset_ins = ins.loc[ins["SK_ID_PREV"].isin(ids_prev), :].drop("SK_ID_CURR", axis=1)
        subset_ccb = ccb.loc[ccb["SK_ID_PREV"].isin(ids_prev), :].drop("SK_ID_CURR", axis=1)
        subset_pos = pos.loc[pos["SK_ID_PREV"].isin(ids_prev), :].drop(["SK_ID_CURR", "Unnamed: 0"], axis=1)
        
        prev_explorer(subset_prev, subset_ins, subset_pos, subset_ccb, variables)