#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:10:57 2020

@author: Alex
"""

import requests
import pandas as pd

url = 'https://ocp7-api.herokuapp.com'

data = pd.read_csv("../../data/output_data/X_test.csv", nrows=10).set_index("SK_ID_CURR").iloc[7,:]

j_data = data.to_json()

#print(j_data)

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}

r = requests.post(url, data=j_data, headers=headers)

print(r, r.text)