import pandas as pd
import numpy as np
import sys
import os
def add_one_hot_feature(data,cat_cols,train_indicate=True):
    for col in cat_cols:
        print(col)
        temp=pd.get_dummies(data[col],drop_first=train_indicate,prefix=col)
        temp.index=data.index
        data=pd.concat([data,temp],axis=1)
        print(data.shape)
    data=data.drop(cat_cols,axis=1)
    print(data.shape)
    return data
def generate_test_onehotdata(data_train_onehot,data_test_onehot):
    return 0




if __name__=="__main__":
    sys.path.append('/Users/jjy/Desktop/kaggle/Home Credit Default Risk/data')
    print(sys.path)
    data_train = pd.read_csv('application_train.csv',
                             index_col='SK_ID_CURR')
    data_test = pd.read_csv('application_test.csv',
                            index_col='SK_ID_CURR')
    data_all=data_train.append(data_test)
    print(data_train.shape,data_test.shape,data_all.shape)
    cat_cols=['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE',
             'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','FLAG_MOBIL',
             'FLAG_EMP_PHONE','FLAG_WORK_PHONE','FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','OCCUPATION_TYPE',
             'WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION',
             'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION','REG_CITY_NOT_LIVE_CITY',
             'REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','ORGANIZATION_TYPE','FONDKAPREMONT_MODE',
             'HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE',
             'FLAG_DOCUMENT_2',
             'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
             'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12',
             'FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17',
             'FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    data_all_onehot=add_one_hot_feature(data_all,cat_cols,train_indicate=True)
    data_train_onehot=data_all_onehot.loc[data_train.index]
    data_test_onehot = data_all_onehot.loc[data_test.index]
    print('all shape',data_all_onehot.shape,'train shape',data_train_onehot.shape,'test shape ',data_test_onehot.shape)
    print('train shape', data_train_onehot.shape, 'test shape ', data_test_onehot.shape)
    data_train_onehot.to_csv('train_one_hot.csv')
    data_test_onehot.to_csv('test_one_hot.csv')