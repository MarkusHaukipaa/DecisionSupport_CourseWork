# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 14:17:00 2025

@author: mhauk
"""

import pandas as pd
import numpy as np
import matplotlib.patches as pathes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy as sc
import sklearn as sk
import sklearn.pipeline as pipeline
import neurokit2 as nk2
import matplotlib.dates as mdates 
import sklearn.ensemble as sk_ensemble
from sklearn.metrics import roc_curve, roc_auc_score

def ROCtest(classifier, clf_name):
    y_pred_prob = classifier.predict_proba(X_test)[:, 1] 
 
    # Compute ROC curve 
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) 
     
    # Compute AUC-ROC 
    auc_roc = roc_auc_score(y_test, y_pred_prob) 
    print(f"{clf_name}: AUC-ROC: {auc_roc}") 
    # Find the elbow point 
    distances = np.sqrt((0 - fpr)**2 + (1 - tpr)**2) 
    elbow_index = np.argmin(distances) 
    elbow_threshold = thresholds[elbow_index] 
     
    # Plotting the ROC curve 
    plt.figure(figsize=(10, 8)) 
    plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.2f}') 
    plt.plot([0, 1], [0, 1], 'r--') 
    plt.scatter(fpr[elbow_index], tpr[elbow_index], marker='o', color='red', 
    label=f'Elbow Point (Threshold = {elbow_threshold:.2f})') 
     
    plt.xlim([0, 1]) 
    plt.ylim([0, 1]) 
    plt.xlabel('False Positive Rate (1 - Specificity)') 
    plt.ylabel('True Positive Rate (Recall)') 
    plt.title(clf_name +'Receiver Operating Characteristic (ROC) Curve with Elbow Point') 
    plt.legend(loc='lower right') 
    plt.show()
    
def Ex1():
    #PLot normalized plot and boxplot
    for i in range(len(features)):
        
        plt.scatter(df.index, df_scaled[features[i]], label= features[i], color = colors)
        plt.title("Normalized plot of feature: " + features[i])
        plt.xlabel("patient number")
        plt.ylabel("Normalized value of the respected dataset")
        plt.show()
        plt.cla()
        
    
        plt.boxplot([df_pos_outcome[features[i]], df_neg_outcome[features[i]], df[features[i]]])
        plt.title("Boxplot plot of feature: " + features[i])
        plt.xlabel("Outcome")
        plt.ylabel("Logical value of the respected dataset")
        plt.xticks([1,2,3],["Positive", "Negative", "All"])
        plt.show()
        plt.cla()
    
   
def Ex2():
    
    
    clf_randomForest = sk_ensemble.RandomForestClassifier(max_depth=5, 
                                                          n_estimators=100,
                                                          min_samples_split=8,
                                                          min_samples_leaf=3, 
                                                          max_features= 'sqrt',
                                                          bootstrap= True)
    
    # clf_logReg = sk.linear_model.LogisticRegression(random_state=42, class_weight='balanced' )
    #testing pipeline solution
    pipeLG = pipeline.Pipeline([('scaler', 
                                         sk.preprocessing.StandardScaler()), 
                                        ('LG', 
                                         sk.linear_model.LogisticRegression())])
    
    # train
    clf_randomForest.fit(X_train, y_train)
    # clf_logReg.fit(X_train, y_train)
    
    # test
    y_predicted_rf = clf_randomForest.predict(X_test)
    #y_predicted_lg =clf_logReg.predict(X_test)
    y_predicted_lg_pipe = pipeLG.set_params(LG__random_state = 42, LG__class_weight = 
                      'balanced').fit(X_train, y_train).predict(X_test)
    
    print("RF accuracy:", sk.metrics.accuracy_score(y_test, y_predicted_rf))
    # print("LG accuracy:", sk.metrics.accuracy_score(y_test, y_predicted_lg))
    print("LG_pipe accuracy:", sk.metrics.accuracy_score(y_test, y_predicted_lg_pipe))
    # Scaled performs more accurately
    
    
    sk.metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predicted_rf)
    plt.title("Random Forest: Predictions")
    plt.show()
    
    """sk.metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predicted_lg)
    plt.title("Logistic Regression: Predictions")
    plt.show()""" # It is suggested to scale the data
    
    sk.metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predicted_lg_pipe,)
    plt.title("Logistic Regression with scaler: Predictions")
    plt.show()
    
    print("Results for the Random Forest:")
    print(sk.metrics.classification_report(y_test, y_predicted_rf))
    print("Balanced accuracy is: ",
          sk.metrics.balanced_accuracy_score(y_test,y_predicted_rf))
    
    
    print("Results for the Logistic Regression:")
    print(sk.metrics.classification_report(y_test, y_predicted_lg_pipe))
    print("Balanced accuracy is: ",
          sk.metrics.balanced_accuracy_score(y_test, y_predicted_lg_pipe))
    
  
    ROCtest(pipeLG, "Logistic Regression")
    ROCtest(clf_randomForest, "Random Forest")
def Ex3():
    
    clf_randomForest = sk_ensemble.RandomForestClassifier(max_depth=5, 
                                                          n_estimators=100,
                                                          min_samples_split=8,
                                                          min_samples_leaf=3, 
                                                          max_features= 'sqrt',
                                                          bootstrap= True,
                                                          class_weight= 'balanced')
    clf_randomForest.fit(X_train, y_train)
    y_predicted_rf = clf_randomForest.predict(X_test)
    print("Balanced RF accuracy :", sk.metrics.accuracy_score(y_test, y_predicted_rf))
    
    sk.metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predicted_rf)
    plt.title("Balanced Random Forest: Predictions")
    plt.show()
    
    print("Results for the balanced Random Forest :")
    print(sk.metrics.classification_report(y_test, y_predicted_rf))
    print("Balanced accuracy is: ",
          sk.metrics.balanced_accuracy_score(y_test,y_predicted_rf))
    ROCtest(clf_randomForest, "Random Forest Balanced")
    
def Ex4():
    
    pipeLG = pipeline.Pipeline([('scaler', 
                                         sk.preprocessing.StandardScaler()), 
                                        ('LG', 
                                         sk.linear_model.LogisticRegression())])
    # Define scoring metrics 
    scoring_metrics = ['accuracy', 'precision', 'recall'] 
    # Perform 5-fold stratified cross-validation with multiple metrics 
    stratified_kfold = sk.model_selection.StratifiedKFold()
    cv_results = sk.model_selection.cross_validate(pipeLG, X_train, y_train, cv=stratified_kfold, 
    scoring=scoring_metrics)
    
    print("test accuracy average: ", np.mean(cv_results["test_accuracy"]),
          "std: ", np.std(cv_results["test_accuracy"]))
    print("test precision average: ", np.mean(cv_results["test_precision"]),
          "std: ", np.std(cv_results["test_precision"]))
    print("test recall average: ", np.mean(cv_results["test_recall"]), "std: ",
          np.std(cv_results["test_recall"]))
    
    
# def main() :D just to make speed it up I made in in global :P
df = pd.read_csv(r"C:\Users\mhauk\OneDrive - TUNI.fi\Opinnot\Kurssit\Decision_support\Week_4\diabetes.csv")
features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
'BMI', 'DiabetesPedigreeFunction','Age']

info = df.describe()

colors = df["Outcome"].map({1: "r", 0: "b"})  #  outcome to colors
scaler = sk.preprocessing.MinMaxScaler()

df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), 
                         columns=features, index=df.index)

df_pos_outcome = df.loc[df["Outcome"]==1]
df_neg_outcome = df.loc[df["Outcome"]==0]


#Ex1()

outcome = df["Outcome"]

X_train, X_test, y_train, y_test =(
    sk.model_selection.train_test_split(df.drop(columns = ["Outcome"]),outcome, test_size=0.3,
                                        random_state=42, stratify=outcome))
#Ex2()
#Ex3()
#Ex4()