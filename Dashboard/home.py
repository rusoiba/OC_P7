#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 13:58:46 2020

@author: Alex
"""
import streamlit as st
from PIL import Image
import pandas as pd


@st.cache()
def load_test_data():
	"""cached function that returns our working examples for the dashboard"""
    return pd.read_csv("../data/output_data/X_test.csv")

def home() : 
	"""principal page of the dashboard"""
    st.markdown("This application provides 3 mains modules :")
    st.markdown("* **The prediction module :** enables you to assess client's liability based on its file")
    st.markdown("* **The explorer module :** enables you to dig deeper into your client informations,"
                " particularly historcial data coming from federal loan bureau and historical"
                " Home Credit's data if available.")
    st.markdown("* **The statistics module** : enables you to explore the database at a macro scale :"
                " understand how variables such as age, sex and income impact probability of repayment")