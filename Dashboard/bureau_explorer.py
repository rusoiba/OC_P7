#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 13:37:52 2020

@author: Alex
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def bureau_explorer(bureau, bb, variables) : 
    """Explorer, displays all bureau and bb related data"""
    # Bureau
    sns.set()
    
    st.header("Summary from Credit Bureau")
    bureau_disp_feat = ['CREDIT_ACTIVE', 'DAYS_CREDIT','AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'CREDIT_TYPE']
    st.dataframe(bureau[bureau_disp_feat])
    feat_request_info = st.selectbox("Select the variable you want to learn about", bureau_disp_feat)
    st.write(variables.loc[variables["Row"]==feat_request_info, "Description"].values[0])
        
    f, axs = plt.subplots(1,2,figsize=(10,5))

    ax1 = axs[0] 
    bureau['CREDIT_ACTIVE'].value_counts().plot.pie(ax=ax1)
    ax1.set_title("Share of Active and Closed ")
    ax1.set_ylabel("")
    
    ax2 = axs[1]
    active_credits = bureau.loc[bureau["CREDIT_ACTIVE"]=="Active", :]
    total_amt = active_credits["AMT_CREDIT_SUM"].sum()
    paid_amt = active_credits["AMT_CREDIT_SUM"].sum() - active_credits["AMT_CREDIT_SUM_DEBT"].sum()
    debt_amt = active_credits["AMT_CREDIT_SUM_DEBT"].sum()
    
    plt.barh(y=0, width = total_amt, label="paid on active loans")
    plt.barh(y=0, left = total_amt, width = debt_amt, label="to remburse on active loans")
    plt.legend()
    
    ax2.set_title("Proportion of total amount that is still pending")
    top_side = ax2.spines["top"]
    right_side = ax2.spines["right"]
    

    right_side.set_visible(False)
    top_side.set_visible(False)

    st.pyplot()
    
    st.header("Financial Behavior")
    
    bb_disp_feat = ["0", "1", "2", "3", "4", "5"]
    bb[bb_disp_feat].sum(axis=0).plot.bar()
    locs, labels = plt.xticks() 
    plt.xticks(locs, ["0", "0-30", "31-60", "61-90", "91-120", "+120"], rotation=45)
    plt.title("All monthly payments, classified by importance of delay (in days)")
    st.pyplot()