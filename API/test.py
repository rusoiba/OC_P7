#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 14:06:10 2020

@author: Alex
"""

import unittest
import requests
import pandas as pd

class TestAPI(unittest.TestCase):

    def test_requesting(self):
        url = 'http://0.0.0.0:80/api/'
        data = pd.read_csv("../../data/output_data/X_test.csv", nrows=5).set_index("SK_ID_CURR").iloc[3,:]
        j_data = data.to_json()
        
        headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
        
        r = requests.post(url, data=j_data, headers=headers)
        print(r)
        print(r.text)
        
        self.assertTrue(r.text != None)

if __name__ == '__main__':
    unittest.main()