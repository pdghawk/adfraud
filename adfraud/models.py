""" Module for ML models 
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

import category_encoders as ce

class HashForest:
    """ model for Hash encoding followed by random forest
    """
    def __init__(self,feats,feats_to_hash):
        self.feats=feats
        self.feats_to_hash = feats_to_hash
        #----------------------------------------
        # build the pipeline for the Hashing and Forest
        # feature encoder
        hasher = ce.hashing.HashingEncoder(n_components=50,cols=self.feats_to_hash)
        # classifier
        clf = RandomForestClassifier(n_estimators=20, max_depth=5,class_weight={0:0.001,1:0.999})
        self.model = Pipeline([('hash',hasher),('forest',clf)])

    def fit(self,x_train,y_train):
        self.model.fit(x_train[self.feats],y_train)

    def train_CV(self,x_train,y_train,param_grid,n_splits=5):
        cv = StratifiedShuffleSplit(n_splits=n_splits,test_size=0.2,random_state=42)
        self.model_cv = GridSearchCV(self.model,param_grid,cv=cv,scoring='roc_auc',verbose=50)
        self.model_cv.fit(x_train[self.feats],y_train)

    def test_auc(self,x_test,y_test):
        preds = self.model_cv.predict_proba(x_test[self.feats])
        fpr, tpr, thresholds = metrics.roc_curve(y_test, preds[:,1])
        auc = metrics.roc_auc_score(y_test, preds[:,1])
        return auc

    @property
    def train_auc(self):
        return self.model_cv.best_score_
