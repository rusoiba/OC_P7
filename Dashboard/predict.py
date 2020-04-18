0#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 11:34:32 2020

@author: Alex
"""
import streamlit as st
import pandas as pd
import numpy as np
import shap
import seaborn as sns
from functions import load_training_data, load_test_data,\
load_variable_description, load_model, compute_tree_explainer

    
def predict(id_curr):
    """returns the elements of prediction page"""
    sns.reset_orig()
    X_train = load_training_data()
    X_test = load_test_data()
    
    lgbm = load_model()
    print("id data", hex(id(X_train)))
    print("id lgbm", hex(id(lgbm)))
    explainer = compute_tree_explainer(lgbm, X_train)
    print("id explainer", hex(id(explainer)))

    
    ids_avail = X_test["SK_ID_CURR"]
    if (ids_avail == id_curr).sum() > 0 : 
        to_analyse = X_test.loc[X_test["SK_ID_CURR"] == id_curr, :].drop(columns=["SK_ID_CURR"])
        shap.initjs()
        
        shap_values = explainer.shap_values(to_analyse, check_additivity=True)
        
        st.subheader("Probability of payment default for loan ID{}".format(id_curr))
        shap.force_plot(explainer.expected_value,
                                 shap_values[0],
                                 to_analyse.round(2),
                                 matplotlib=True,
                                 link="logit"
                                 )
    
        st.pyplot(bbox_inches='tight',
                  dpi=500,pad_inches=0)
        
        shap_named = pd.Series(np.copy(shap_values[0]), index=X_test.drop(columns=["SK_ID_CURR"]).columns)
        
        most_imp_feat = abs(shap_named).sort_values(ascending=False).head(10).index
        displ_feat = shap_named[most_imp_feat].sort_values()
        variables = load_variable_description()
        
        info_feat = st.selectbox("Select the variable you want to know more about", displ_feat.index)
        st.write(info_feat)
        st.write(to_analyse.loc[:,info_feat].values[0].round(2), variables.loc[variables["Row"]==info_feat, "Description"].values[0])
                
    else : 
        st.error("Solve error in the sidebar before accessing this module")