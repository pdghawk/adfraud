""" Module for ML models
"""
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.pipeline import Pipeline

import category_encoders as ce

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from . import error as err

class HashForest:
    """ model for Hash encoding followed by random forest
    """
    def __init__(self,feats,feats_to_hash):
        """ Initialize the model
        Args:
            - feats: list of features (column names as strings) to include in the model
            - feats_to_hash: list of features (column names as strings) to hash encode
        """
        self.feats=feats
        self.feats_to_hash = feats_to_hash
        #----------------------------------------
        # build the pipeline for the Hashing and Forest
        # feature encoder
        hasher = ce.hashing.HashingEncoder(n_components=50,cols=self.feats_to_hash)
        # classifier
        clf = RandomForestClassifier(n_estimators=20, max_depth=5,class_weight={0:0.001,1:0.999})
        self.model = Pipeline([('hash',hasher),('forest',clf)])
        self.validated = False

    def fit(self,x_train,y_train):
        """ fit the model """
        self.model.fit(x_train[self.feats],y_train)

    def train_CV(self,x_train,y_train,param_grid,n_splits=5):
        """ train the model with Criss Validation

        Performs a stratified split of the training data into train and validation
        partitions (keeping ratio of class type similar in both sets).
        Scoring in cross validation is set to be area under ROC.

        Args:
            - x_train: training data (features, can contain all columns)
            - y_train: training labels
            - param_grid: grid of parameters to scan over
            - n_splits: (optional, default=5) how many splits to perform in cross validation
        """
        cv = StratifiedShuffleSplit(n_splits=n_splits,test_size=0.2,random_state=42)
        self.model_cv = GridSearchCV(self.model,param_grid,cv=cv,scoring='roc_auc',verbose=10)
        self.model_cv.fit(x_train[self.feats],y_train)
        self.validated=True

    def test_auc(self,x_test,y_test):
        """ test the model Area Under Curve (ROC)

        Get the AUC on test data for the cross_validated model

        """
        if self.validated:
            preds = self.model_cv.predict_proba(x_test[self.feats])
            auc = metrics.roc_auc_score(y_test, preds[:,1])
            return auc
        else:
            raise err.NotYetFittedModelError('You should train the model with method train_CV first \n')

    @property
    def train_auc(self):
        """ get AUC on training data """
        if self.validated:
            return self.model_cv.best_score_
        else:
            raise err.NotYetFittedModelError('You should train the model with method train_CV first \n')

    def plot_roc_acc(self,x_test,y_test):
        """ plt ROC and accuracy curves on test set

        Args:
         - x_test: test data (features)
         - y_test: test labels

        """
        if self.validated:
            preds = self.model_cv.predict_proba(x_test[self.feats])
            fpr, tpr, thresholds = metrics.roc_curve(y_test, preds[:,1])
            auc = metrics.roc_auc_score(y_test, preds[:,1])

            # get accuracy of class0 and class1 predictions at different thresholds
            acc_0 = []
            acc_1 = []
            for ind,th in enumerate(thresholds):
                binary_preds = pred_from_prob(preds,th)
                acc_1.append(accuracy_score(y_test[np.where(y_test==1)],binary_preds[np.where(y_test==1)]))
                acc_0.append(accuracy_score(y_test[np.where(y_test==0)],binary_preds[np.where(y_test==0)]))

            # set up plotting parameters
            sns.set()
            mpl.rcParams['figure.figsize']=[15.0,5.0]
            mpl.rcParams['lines.linewidth']=2.0
            mpl.rcParams['xtick.labelsize']=13
            mpl.rcParams['ytick.labelsize']=13
            mpl.rcParams['axes.labelsize']=15
            mpl.rcParams['axes.labelweight']='heavy'
            mpl.rcParams['axes.titlesize']=18
            mpl.rcParams['axes.titleweight']='heavy'
            mpl.rcParams['legend.fontsize']=12

            # build the plots- ROC in subplot 1, accuracy in subplot 2
            plt.subplot(1,2,1)
            plt.plot(fpr,tpr,'r*', label='roc auc={:.4f}'.format(auc))
            plt.plot(np.linspace(0,1,20),np.linspace(0,1,20),'k--')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('ROC')
            plt.legend(frameon=False)
            plt.xlim([0.0,1.0])
            plt.ylim([0.0,1.0])

            plt.subplot(1,2,2)
            plt.plot(thresholds,acc_0,'go--',label='cases: 0',alpha=0.3)
            plt.plot(thresholds,acc_1,'yo--',label='cases: 1')
            plt.plot(thresholds[::5],fpr[::5],'c*',alpha=0.2,label='fpr')

            plt.xlim([0,1])
            plt.xlim([0,1])

            plt.xlabel('threshold')
            plt.ylabel('accuracy')
            plt.title('accuracy')

            plt.legend(frameon=False)

            plt.show()
            return None
        else:
            raise err.NotYetFittedModelError('You should train the model with method train_CV first \n')


def pred_from_prob(a,threshold):
    """ turn two-class percentage prediction to binary
    Args:
    - a: probabilities; shape=(n_samples,2)
    - threshold: probability above which class 1 will be counted as 1
    """
    bin_preds = np.zeros((np.size(a,0),))
    bin_preds[np.where(a[:,1]>threshold)]=1.0
    return bin_preds
