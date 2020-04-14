#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:22:43 2020

@author: Alex
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def prev_explorer(prev, ins, pos, ccb, variables) : 
    st.header("Previous Applications")
    
    prev_disp_feat = ['NAME_CONTRACT_TYPE', 'AMT_ANNUITY',
       'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
       'RATE_DOWN_PAYMENT', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE',
       'CODE_REJECT_REASON', 'NAME_CLIENT_TYPE', 'NAME_PORTFOLIO',
       'CHANNEL_TYPE','CNT_PAYMENT', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION']
    st.dataframe(prev[prev_disp_feat])
    feat_request_info = st.selectbox("Select the variable you want to learn about", prev_disp_feat)
    st.info(variables.loc[variables["Row"]==feat_request_info, "Description"].values[0])
    
    st.subheader("Application Analysis")
    prev_ids = st.multiselect("Select the application you want to see the time series", list(prev.index))
    plt.scatter(x=(ins.loc[ins["SK_ID_PREV"].isin(prev_ids), "DAYS_INSTALMENT"]).astype(int),
             y=ins.loc[ins["SK_ID_PREV"].isin(prev_ids), "AMT_INSTALMENT"].astype(int))
    
    st.pyplot()
    
    #st.dataframe(ins.loc[ins["SK_ID_PREV"].isin(prev_ids), ["DAYS_INSTALMENT", "AMT_INSTALMENT"]])
    
    
    st.subheader("Point of Sales and Cash Loans")
    st.write(pos.sum(axis=0)["SK_DPD"], variables.loc[(variables["Row"]=="SK_DPD") & (variables["Table"]=="POS_CASH_balance.csv"), "Description"].values[0])
    st.write(pos.sum(axis=0)["SK_DPD_DEF"], variables.loc[(variables["Row"]=="SK_DPD_DEF") & (variables["Table"]=="POS_CASH_balance.csv"), "Description"].values[0])
    
    
    st.subheader("Credit Card Balance")
    #st.dataframe(ccb)
    plt.scatter(x=ccb["MONTHS_BALANCE"], y=ccb["AMT_BALANCE"])
    plt.title("Balance during the previous credit and mean balance (red)")
    plt.ylabel("Amount on credit card")
    plt.xlabel("Timeline in months (0 is application month)")
    plt.axhline(y=ccb["AMT_BALANCE"].mean(), linestyle="--", color="r", lw=1)
    st.pyplot()
