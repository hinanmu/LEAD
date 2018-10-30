#@Time      :2018/10/17 16:44
#@Author    :zhounan
# @FileName: bayesian_network.py

import pandas as pd
import numpy as np
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, BicScore

def build_structure(data):
    df = pd.DataFrame(data)
    est = HillClimbSearch(df, scoring_method=BicScore(df))
    model = est.estimate()
    DAG = np.zeros((data.shape[1], data.shape[1]), np.int64)

    for edge in model.edges():
        DAG[edge[0], edge[1]] = 1

    np.save('dataset/DAG.npy', DAG)
    return DAG

if __name__=='__main__':
    errors = np.load('dataset/errors.npy')
    build_structure(errors)