import pandas as pd
import lightgbm as lgm
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from evolutionary_search import EvolutionaryAlgorithmSearchCV
class model:
    def __init__(self):
        self.train_x,self.test_x,self.train_y,self.test_y,self.pred_data=self.read_data()
    def read_data(self):
        train_data = pd.read_csv('data/train_one_hot.csv', index_col='SK_ID_CURR')
        pred_data = pd.read_csv('data/test_one_hot.csv', index_col='SK_ID_CURR')
        train_x, test_x, train_y, test_y = train_test_split(train_data.drop('TARGET', axis=1), train_data['TARGET'],
                                                            test_size=0.2)
        train_x = train_x.fillna(0)
        test_x = test_x.fillna(0)
        pred_data = pred_data.fillna(0).drop('TARGET', axis=1)
        return train_x,test_x,train_y,test_y,pred_data

def grid_search(clf,params,scoring,search_method='ev'):
    if search_method == 'grid':
        gs=GridSearchCV(clf,param_grid=params,scoring=scoring,cv=2,refit=True,n_jobs=2,verbose=2)
    elif search_method == 'ev':
        gs=EvolutionaryAlgorithmSearchCV(clf,params=params,scoring=scoring,cv=2,refit=True,n_jobs=2,verbose=2,
                                        population_size=8,
                                        gene_mutation_prob=0.10,
                                        gene_crossover_prob=0.5,
                                        tournament_size=3,
                                        generations_number=4
                                        )
    else:
        gs=clf

    gs.fit(train_x,train_y)
    return gs
def get_pred_data(gs,pred_data):
    res=pd.DataFrame(gs.predict_proba(pred_data)[:,1])

    res.index=pred_data.index
    res.index.name='SK_ID_CURR'
    res.columns=['TARGET']
    return res
def get_test_auc(clf,test_x,test_y):
    pred = clf.predict_proba(test_x)
    print(roc_auc_score(test_y, pred[:, 1]))

if __name__=='__main__':
    train_data=pd.read_csv('data/train_one_hot.csv',index_col='SK_ID_CURR')
    train_x,test_x,train_y,test_y=train_test_split(train_data.drop('TARGET',axis=1),train_data['TARGET'],test_size=0.2)
    train_x=train_x.fillna(0)
    test_x=test_x.fillna(0)
    clf=GradientBoostingClassifier()
    #clf2=lgm.LGBMClassifier()
    scoring='roc_auc'
    params_gbdt={'learning_rate':[0.001,0.005,0.01],
            'max_depth':[3,5,8,10],
            'subsample':[0.7,0.8,0.9,1],
            'n_estimators':[100,200,500]}


    gs=grid_search(clf,params_gbdt,scoring,search_method='ev')
    get_test_auc(gs,test_x,test_y)
    pred_data=pd.read_csv('data/test_one_hot.csv',index_col='SK_ID_CURR')
    print(test_x.shape,pred_data.shape)
    pred_data = pred_data.fillna(0).drop('TARGET',axis=1)
    res=get_pred_data(gs,pred_data)
    res.to_csv('submission.csv')


