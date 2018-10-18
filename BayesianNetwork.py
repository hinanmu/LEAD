#@Time      :2018/10/17 16:44
#@Author    :zhounan
# @FileName: bayesian_network.py

import pandas as pd
import numpy as np
from pgmpy.estimators import ExhaustiveSearch

def build_structure(data):
    df = pd.DataFrame(data)
    est = ExhaustiveSearch(df)
    model = est.estimate()
    file = open('prepare_data/DAG.txt','w')
    file.write(str(model.edges()))
    file.close()
    print(model.edges())
    return model.edges

