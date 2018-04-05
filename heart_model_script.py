# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 03:53:38 2018

@author: Vasudev
"""

from sklearn.externals import joblib
import numpy as np
import sys

svc_classifier=joblib.load('heart_linear_classifier_file')
a=np.array([ 54. ,   1. ,   3. , 120. , 258. ,   0. ,   2. , 147. ,   0. ,
          0.4,   2. ])
#test_point_input=sys.argv[1]
test_point=a
result=svc_classifier.predict_proba(test_point)
print(result[0][0])
