#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 22:46:27 2020

@author: Alex
"""

from flask import Flask, request, redirect, url_for, flash, jsonify
from joblib import load
import numpy as np
import os
import pandas as pd
import json
from transformers import *
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

estimator_file = './estimator.joblib'
estimator = load(estimator_file)

dtypes_data = pd.read_csv("./X_train_sample.csv", index_col=0).drop("SK_ID_CURR", axis=1)
dtypes_dict = dict(dtypes_data.dtypes)

app = Flask(__name__)

@app.route('/', methods=['POST'])
def makecalc():
    j_data = request.get_json()

    data = pd.DataFrame(j_data, index=["to_predict"]).astype(dtypes_dict)
    prediction = estimator.predict_proba(data)

    encodedNumpyData = json.dumps(prediction, cls=NumpyArrayEncoder)

    return encodedNumpyData

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
    #host='127.0.0.1', port=5000