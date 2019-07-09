Adfraud
==========

A personal project to look at ad fraud data, taken from the Kaggle competition:

TalkingData AdTracking Fraud Detection Challenge
(https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)

Notebooks exploring the data can be found in notebooks/. notebooks/exploratory.ipynb
in particular contains a first look through the data.

The package /adfraud contains some modules in early development for building
ML models to study the data.

see notebooks/multimodel.ipynb to see testing of different models using the
adfraud package.

Simple exploration of the data was used to quickly analyze which features were
likely to be most important. Utilizing these findings with hash encoding and a
random forest model, with only minimal hyperparameter optimization yields an area
under the yield curve of 0.957 - on the test set where training and testing is
only performed on a small subsample of all data.


.. image :: img/example.png


Training on the full dataset and with further optimization of the model would likely
lead to an improved performance. 
