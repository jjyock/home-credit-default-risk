import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class feature_explore:
    def __init__(self,label_col):
        self.data = self.get_data()
        self.data_p=self.data[self.data[label_col]==1]
        self.data_n=self.data[self.data[label_col]==0]
    def get_data(self):
        data = pd.read_csv('data/application_train.csv',index_col='SK_ID_CURR',nrows=1000)
        return data




if __name__ == '__main__':
    cat_col=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR']
    F=feature_explore('TARGET')
    print(F.data_p.head())