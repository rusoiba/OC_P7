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

@st.cache()
def load_raw_train_data(): 
    return pd.read_csv("../data/input_data/application_train.csv")


def statistics() : 
    data = load_raw_train_data().set_index("SK_ID_CURR")
    #st.dataframe(data.head().dtypes)
    columns = list(data.columns)
    columns.remove("TARGET")
    
    var = st.selectbox("wich var", columns)
    if data[var].dtypes == int:
        agg_data = data.groupby(var)["TARGET"].agg(["size", "mean"])
        #st.dataframe(agg_data.loc[agg_data["size"]>100, "mean"])
        agg_data.loc[agg_data["size"]>100, "mean"].plot.bar()
        plt.title("Repayment Difficulties as a function of {}".format(var))
        plt.ylabel("Probability of default")
        st.pyplot(bbox_inches='tight', dpi=500,pad_inches=1)
        
    elif data[var].dtypes == float:
        max_var = data[var].max()
        min_var = data[var].min()
        
        
        x = pd.cut(data[var], bins=np.arange(min_var, max_var, (max_var-min_var)/10)).astype(str)
        y = data.groupby(x)["TARGET"].agg(["size", "mean"])
        plt.title("Repayment Difficulties as a function of {}".format(var))
        plt.ylabel("Probability of default")
        y.loc[y["size"]>100, "mean"].plot.bar()
       # plt.bar(x=x, height=y)
        st.pyplot(bbox_inches='tight', dpi=500,pad_inches=1)
        
    else :
        agg_data = data.groupby(var)["TARGET"].agg(["size", "mean"])
        agg_data.loc[agg_data["size"]>100, "mean"].plot.bar()
        plt.title("Repayment Difficulties as a function of {}".format(var))
        plt.ylabel("Probability of default")
        st.pyplot(bbox_inches='tight', dpi=500,pad_inches=1)
