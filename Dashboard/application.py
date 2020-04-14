d#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:09:35 2020

@author: Alex
"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import time
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from joblib import load

from home import *
from explorer import *
from predict import *
from statistics import *

X_test = load_test_data()
    
option = st.sidebar.selectbox(
    'Module Selection',
     ["Home", "Predict", "Explorer", "Statistics"])

id_curr = int(st.sidebar.text_input("Select your client current loan ID :", "100028"))
ids_avail = X_test["SK_ID_CURR"]
if (ids_avail == id_curr).sum() > 0 : 
    st.sidebar.success("Saved : ID{}".format(id_curr))
else :
    image = Image.open("../data/image_data/application_image_very_small.png")
    st.sidebar.error("Cannot find ID {} in Database. You can find \
             this ID on the loan application form :".format(id_curr))
    st.sidebar.image(image, use_column_width=True)


st.title(option)

if option =="Home" : 
    home()
    
elif option == "Predict" : 
    predict(id_curr)
    
elif option == "Explorer" : 
    explorer(id_curr)
    
elif option == "Statistics" : 
    statistics()