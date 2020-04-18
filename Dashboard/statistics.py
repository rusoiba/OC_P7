#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:02:01 2020

@author: Alex
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import load_raw_train_data, load_raw_test_data

def statistics(id_curr) : 
    """compute visual object and responsible for display"""
    raw_data = load_raw_train_data()
    data = raw_data.copy().set_index("SK_ID_CURR")
    
    X_test = load_raw_test_data()
    to_analyse = X_test.loc[X_test["SK_ID_CURR"] == id_curr, :].drop(columns=["SK_ID_CURR"])

    #st.dataframe(data.head().dtypes)
    columns = list(data.columns)
    columns.remove("TARGET")
    
    var = st.selectbox("Which variable do you want to analyse ?", columns)
    if data[var].dtypes == int:
        agg_data = data.groupby(var)["TARGET"].agg(["size", "mean", "median"])
        #st.dataframe(agg_data.loc[agg_data["size"]>100, "mean"])
        agg_data.loc[agg_data["size"]>50, "mean"].plot.barh()
        plt.title("Repayment Difficulties as a function of {}".format(var))
        plt.xlabel("Probability of default")
        st.pyplot(bbox_inches='tight', dpi=500,pad_inches=1)
        
        st.write("Your client has value", to_analyse[var].values[0], "for variable", var)
        
    elif data[var].dtypes == float:
        max_var = data[var].max()
        min_var = data[var].min()
        
        
        x = pd.cut(data[var], bins=np.arange(min_var, max_var, (max_var-min_var)/10)).astype(str)
        y = data.groupby(x)["TARGET"].agg(["size", "mean", "median"])
        plt.title("Repayment Difficulties as a function of {}".format(var))
        plt.xlabel("Probability of default")
        y.loc[y["size"]>50,  "mean"].plot.barh()

       # plt.bar(x=x, height=y)
        st.pyplot(bbox_inches='tight', dpi=500,pad_inches=1)
        
        st.write("Your client has value", to_analyse[var].values[0], "for variable", var)

        
    else :
        agg_data = data.groupby(var)["TARGET"].agg(["size", "mean", "median"])
        agg_data.loc[agg_data["size"]>50,  "mean"].plot.barh()
        plt.title("Repayment Difficulties as a function of {}".format(var))
        plt.xlabel("Probability of default")
        st.pyplot(bbox_inches='tight', dpi=500,pad_inches=1)
        
        st.write("Your client has value", to_analyse[var].values[0], "for variable", var)

