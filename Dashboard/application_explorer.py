#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:01:11 2020

@author: Alex
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from functions import load_variable_description, load_raw_test_data

def application_explorer(id_curr) : 
        """compute and display all informations related to the client current application"""
        test_data = load_raw_test_data()
        test_data["AGE"] = -test_data['DAYS_BIRTH']/365
        test_data["YEARS_EMPLOYED"] = -test_data['DAYS_EMPLOYED']/365
        test_data = test_data.loc[test_data["SK_ID_CURR"]==id_curr,:].round(2).set_index("SK_ID_CURR")
        loan_feat = ['NAME_CONTRACT_TYPE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']
        documents = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
        personnal = ['CODE_GENDER', 'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY', 'NAME_TYPE_SUITE', 'NAME_HOUSING_TYPE', "AGE",
        'FLAG_PHONE', 'FLAG_EMAIL', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
        family = ['CNT_CHILDREN', 'NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS']
        work = ['AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'YEARS_EMPLOYED', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE']
        
        st.subheader("Application")
        st.table(test_data[loan_feat].T)
        
        st.subheader("More informations")
        option = st.selectbox("WHat subject are you interested in ?",
                     ["Legal Documents", "Personnal Data", "Family/Environment", "Work and Incomes"])
        
        corresp = {"Legal Documents" : documents,
                   "Personnal Data": personnal,
                   "Family/Environment" : family,
                   "Work and Incomes" : work}
        
        st.table(test_data[corresp[option]].T)