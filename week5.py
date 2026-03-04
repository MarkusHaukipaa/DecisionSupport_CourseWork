# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 16:15:26 2025

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
import sklearn.inspection as sk_inspection
from sklearn import tree
import graphviz 

#Ex1 + 5
def Ex1():
    pipeLG = pipeline.Pipeline([('scaler', 
                                         sk.preprocessing.StandardScaler()), 
                                        ('LG', 
                                         sk.linear_model.LogisticRegression())])
    
    
    
    pipeLG.set_params(LG__random_state = 42, LG__class_weight = 'balanced').fit(X_train, y_train)
    
    importance = pipeLG['LG'].coef_[0]
    plt.bar([x for x in range(len(importance))], importance, tick_label=features) 

    plt.title("Feature Importance chart LG")
    plt.xticks(rotation = 78)
    plt.ylabel("importance %")
    plt.show()
    
    fig_train, (ax1,ax2,ax3,ax4) = plt.subplots(4, 2, figsize=(20, 12)) 
    sk_inspection.PartialDependenceDisplay.from_estimator(pipeLG, X_train,X_train.columns, 
    ax=[ax1,ax2,ax3,ax4])
        
    
def Ex2():
    clf_decision_tree = tree.DecisionTreeClassifier(max_depth=5,
                                                          min_samples_split=8,
                                                          min_samples_leaf=3,
                                                          class_weight= "balanced")
    
    clf_decision_tree.fit(X_train, y_train)
    
    y_predicted_dt = clf_decision_tree.predict(X_test)
    sk.metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_predicted_dt)
    plt.title("Decision tree: Predictions")
    plt.show()
    print(sk.metrics.classification_report(y_test, y_predicted_dt))
    
    dot_data = tree.export_graphviz(clf_decision_tree, 
                         feature_names= features,  
                         max_depth= 5,
                         class_names= ["Non-diabetic", "Diabetic"],  
                         filled=True, rounded=True,  
                         special_characters=True)  
    
    graph = graphviz.Source(dot_data)  
    graph.view()
    
def Ex3():
    clf_randomForest = sk_ensemble.RandomForestClassifier(max_depth=5, 
                                                          n_estimators=100,
                                                          min_samples_split=8,
                                                          min_samples_leaf=3, 
                                                          bootstrap= True)
    clf_randomForest.fit(X_train, y_train)
    
    plt.bar(features,clf_randomForest.feature_importances_)
    plt.title("Feature related importance RF")
    plt.xticks(rotation = 78)
    plt.ylabel("Importance %")
    plt.show()
    plt.cla()
    
    #Ex4
    result = sk_inspection.permutation_importance(clf_randomForest,
                                                   X_train,y_train,
                                                   n_repeats=30,
                                                   random_state=42)
    sorted_idx = result.importances_mean.argsort()
    feature_importance_df = pd.DataFrame({ 
        'Feature': np.array(features)[sorted_idx],
        'Importance_Mean': result.importances_mean[sorted_idx], 
        'Importance_Std': result.importances_std[sorted_idx] 
    }) 
    plt.bar(feature_importance_df["Feature"],
            feature_importance_df["Importance_Mean"])
    plt.title("Permutation importance RF")
    plt.xticks(rotation = 78)
    plt.ylabel("Importance")
    plt.show()
    
    

    
# def main() :D just to make speed it up I made in in global :P
df = pd.read_csv(r"C:\Users\mhauk\OneDrive - TUNI.fi\Opinnot\Kurssit\Decision_support\Week_5\diabetes.csv")
features=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
'BMI', 'DiabetesPedigreeFunction','Age']

info = df.describe()

colors = df["Outcome"].map({1: "r", 0: "b"})  #  outcome to colors

df_pos_outcome = df.loc[df["Outcome"]==1]
df_neg_outcome = df.loc[df["Outcome"]==0]

outcome = df["Outcome"]

X_train, X_test, y_train, y_test =(
    sk.model_selection.train_test_split(df.drop(columns = ["Outcome"]),outcome, test_size=0.3,
                                        random_state=42, stratify=outcome))


Ex1() # & Ex5

Ex2()
Ex3() #& Ex4

"""pd_results = partial_dependence(clf_randomForest, X_testx, features=0, kind="average",
                                grid_resolution=5)

sk_inspection.PartialDependenceDisplay(pd_results, features, feature)"""