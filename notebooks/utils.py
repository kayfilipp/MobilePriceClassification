import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestClassifier

def getFeatureImportance(df,target):

    X_train, y_train = df.drop([target],axis=1), df[target]

    # Intializing Feature Selector
    feature_selector = RandomForestClassifier()
    feature_selector.fit(X_train,y_train)

    # Potting Data
    features = X_train.columns
    feature_imps = feature_selector.feature_importances_
    mean_imp = feature_imps.mean()

    # Plotting
    plt.figure(figsize=(20,8))
    plt.title("Feature Importnace",fontsize=15)
    plt.bar(features,feature_imps,color="black",label="Not Important")
    plt.bar(features[feature_imps>mean_imp],feature_imps[feature_imps>mean_imp],color="r",label="Important")
    plt.axhline(mean_imp,color="k",linestyle="dashed")
    plt.xlabel("Features",fontsize=12)
    plt.ylabel("Importace Score",fontsize=12)
    plt.legend(fontsize=22)
    plt.show()
    
def sdize(x,mu,sigma):
    return (x-mu)/sigma 
    
def standardScale(frame,features):

    for feature in features:
        mu = frame[feature].mean()
        sigma = frame[feature].std()
        
        frame = frame[feature].apply(lambda x: sdize(x,mu,sigma))
    return frame 