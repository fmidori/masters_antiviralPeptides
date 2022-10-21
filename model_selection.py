import numpy as np
#import pandas as pd
from preprocess import gen_features,create_X_and_y,merge_features,apply_pca,smote,preprocess
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score,recall_score,precision_score,accuracy_score
from sklearn.pipeline import Pipeline
#from imblearn.over_sampling import SMOTE
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.pipeline import Pipeline


# Test One vs Rest Classifier with standard parameters. Returns report of results. 
# algo = the name of classifier choosen. 
def test_with_std(algo,X_train,y_train,X_test,y_test):
    if algo == SGDClassifier:
        clf = OneVsRestClassifier(SGDClassifier(loss="perceptron",random_state=42, class_weight='balanced', max_iter=2000)).fit(X_train,y_train)
    elif algo == LogisticRegression:
        clf = LogisticRegression(solver='liblinear',random_state=42, class_weight='balanced').fit(X_train,y_train)
    else:
        clf = OneVsRestClassifier(algo(random_state=42, class_weight='balanced')).fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    #report = f1_score(y_test, y_pred, average='weighted')
    #print(algo)
    return report

# Create pipeline with PCA step before applyng classifier. With PCA 95% variance (can be changed)
# algo = the name of classifier choosen. 
# Returns pipeline
def create_pipeline_with_pca(algo):
    pca = PCA(n_components = 0.95) # 95% variance 
    #scaler = StandardScaler()
    if "RF" in algo:
        clf = OneVsRestClassifier(RandomForestClassifier())
        pipe = Pipeline(steps=[('pca',pca),('rf',clf)])
    elif "SVM" in algo:
        clf = OneVsRestClassifier(SVC())
        pipe = Pipeline(steps=[("pca", pca), ("svm", clf)])
    return pipe

# Performs GridSearchCV with pipeline (includes PCA) with 3-fold cross validation.
# Classifiers available: RF, SVM (poly and rbf).
# Returns best classifier found after GridSearch 
def grid_search_with_pipe(algo,pipe,X_train,y_train,vector_size):
    #pca_comp = [5,15,20]
    #if vector_size > 20:
    #    for i in range(50, vector_size, 50):
    #        pca_comp.append(i)

    if algo == "RF_pca":
        parameters = {'pca__random_state': [29],
                      'rf__estimator__n_estimators': [50,100,150,200,250], 'rf__estimator__max_features': ['sqrt', 'log2', None],
                      'rf__estimator__random_state': [29], 'rf__estimator__class_weight': ['balanced'], 
                      'rf__estimator__min_samples_leaf': [1,10,50,100], 'rf__estimator__max_depth': [1,5,10,20,50,None] }
    elif algo == "SVM_rbf_pca":
        parameters = {'pca__random_state': [29],
                      'svm__estimator__kernel': ['rbf'] , 'svm__estimator__gamma': [1,0.1,0.01,0.001], 
                      'svm__estimator__C': [0.1,1,10,100],
                      'svm__estimator__class_weight': ['balanced'],'svm__estimator__random_state': [29]}
    elif algo == "SVM_poly_pca":
        parameters = {'pca__random_state': [29],
                      'svm__estimator__kernel': ['poly'] , 'svm__estimator__gamma': [1,0.1,0.01,0.001], 
                      'svm__estimator__C': [0.1,1,10,100], 'svm__estimator__degree': [0,1,2,3,4,5,6], 
                      'svm__estimator__class_weight': ['balanced'],'svm__estimator__random_state': [29]}
    #parameters['pca__n_components']=pca_comp 
    print(parameters)
    gs = GridSearchCV(pipe, parameters,verbose=1, cv=3, scoring='accuracy',n_jobs=8).fit(X_train,y_train)
    return gs.best_estimator_

# Performs GridSearchCV without pipeline (does not includes PCA) with 3-fold cross validation.
# Classifiers available: RF, SVM (poly and rbf).
# Returns best classifier found after GridSearch
def grid_search(X_train,y_train,algo):
    if algo == "RF":
        parameters = {'estimator__n_estimators': [50,100,150,200,250], 'estimator__max_features': ['sqrt', 'log2', None],
                    'estimator__random_state': [29], 'estimator__class_weight': ['balanced'], 
                    'estimator__min_samples_leaf': [1,10,50,100], 'estimator__max_depth': [1,5,10,20,50,None] }
        clf = OneVsRestClassifier(RandomForestClassifier())
    elif algo == "SVM_rbf":
        parameters = {'estimator__kernel': ['rbf'] , 'estimator__gamma': [1,0.1,0.01,0.001], 'estimator__C': [0.1,1,10,100],
                      'estimator__class_weight': ['balanced'],'estimator__random_state': [29]}
        clf = OneVsRestClassifier(SVC())
    elif algo == "SVM_poly":
        parameters = {'estimator__kernel': ['poly'] , 'estimator__gamma': [1,0.1,0.01,0.001], 'estimator__C': [0.1,1,10,100],
                    'estimator__degree': [0,1,2,3,4,5,6], 'estimator__class_weight': ['balanced'],'estimator__random_state': [29]}
        clf = OneVsRestClassifier(SVC())
    elif algo == "SVM_sig":
        parameters = {'estimator__kernel': ['sigmoid'] , 'estimator__gamma': [1,0.1,0.01,0.001], 'estimator__C': [0.1,1,10,100],
                      'estimator__class_weight': ['balanced'],'estimator__random_state': [29]}
        clf = OneVsRestClassifier(SVC())
    gs = GridSearchCV(clf, parameters, verbose=1, cv=3, scoring='accuracy',n_jobs=15).fit(X_train,y_train) #scoring=f1_macro
    return gs.best_estimator_

# Apply best estimator in Test set 
# Returns predictions for Test set
def apply_best_estimator(best_estimator,X_test):
    clf = best_estimator
    y_pred = clf.predict(X_test)
    return y_pred

def class_report(y_test,y_pred):
    report = classification_report(y_test, y_pred)
    return report

# Choose metric to report results
def show_report(metric,y_test,y_pred):
    if metric == "class_report":
        report = classification_report(y_test, y_pred)
    elif metric == 'recall':
        report = recall_score(y_test, y_pred, average='macro')
    elif metric == 'f1_weighted':
        report = f1_score(y_test, y_pred, average='weighted')
    elif metric == 'f1_macro':
        report = f1_score(y_test, y_pred, average='macro')
    elif metric == 'precision':
        report = precision_score(y_test, y_pred, average='macro')
    elif metric == 'accuracy':
        report = accuracy_score(y_test, y_pred)
    return report 


# main function to perform model selection
def main(algo,name_of_out):
    mode = 'smote' # mode can be smote or not_smote
    out = open(name_of_out, "w") # opens an output file 
    out.write("Feature\t" + "Classifier\t" + "Mode\t" + "f1_weighted\t" + "f1_macro\t" + "recall\t" + "precision\t" + "accuracy" + '\n')

    # list of features to convert data
    for feature in  ["AAC+DPC",        
                    "AAC+PCP",
                    "PAAC+PCP","PAAC+DPC",
                    "CTriad+PCP",
                    "AAC",
                    "CTDC+CTDD+CTDT+AAC","CTDC+CTDD+CTDT+PAAC","CTDC+CTDD+CTDT+PCP",
                    "PAAC","DPC","CTriad","PCP","CTDC+CTDD+CTDT"]:
        if "+" in feature:
            X,y = merge_features(feature,3)
        else:  
            X,y = create_X_and_y(feature + ".tsv",3)
        
        vector_size = int(len(X[0]))
        print("X size:", len(X))
        print("X vector size:", len(X[0]))
        print("y label counts:",np.unique(y,return_counts=True))

        # Choose between SVM or RF
        if "SVM" in algo:
            X_train,X_test,y_train,y_test = preprocess(X,y,"yes","yes") #first is scaled, second is pca 
        elif "RF" in  algo:
            X_train,X_test,y_train,y_test = preprocess(X,y,"yes","yes") #first is scaled, second is pca 
        
        # Apply GScv to find best estimator
        best = grid_search(X_train,y_train,algo)
        print(best)
        y_pred = apply_best_estimator(best,X_test)

        # report results in different metrics
        f1_weighted = str(show_report("f1_weighted",y_test,y_pred))
        f1_macro = str(show_report("f1_macro",y_test,y_pred))
        recall = str(show_report("recall",y_test,y_pred))
        precision = str(show_report("precision",y_test,y_pred))
        accuracy = str(show_report("accuracy",y_test,y_pred))
        print(feature + '\n' + show_report("class_report",y_test,y_pred))

        out.write(str(feature) + '\t' + str(algo) + '\t' + mode + '\t' + f1_weighted + '\t' + f1_macro + '\t' + recall + '\t' + precision + '\t' + accuracy + '\n'  )




            #print(class_report(y_test,y_pred))
            #print(best.predict_proba(X_test))
            #cm = multilabel_confusion_matrix(y_test, y_pred)
            #print(cm)           
            
            
            
             # DecisionTreeClassifier, LogisticRegression,
                     #SGDClassifier ]: #MLPClassifier, LinearSVC, ]:
            #report = test_with_std(algo,X_train,y_train,X_test,y_test)

            #out.write(str(feature) + '\t' + str(algo).split('.')[3].split("'")[0] + '\n'  )
            #out.write(report + '\n')


            #out.write(str(feature) + '\t' + str(algo).split('.')[3].split("'")[0] + '\t' + str(report) + '\n'  )
            #out.write(str(report) + '\n')


        #best_estimator = grid_search(X_train,y_train,"RF")
        #y_pred = apply_best_estimator(best_estimator,X_test)
        #report = class_report(y_test,y_pred)
        #print(report)

if __name__ == "__main__":
    #main("RF","GS_RF_scaled_pca.txt")
    #X,y = create_X_and_y("AAC" + ".tsv",3)
    #pca = PCA(n_components = 0.95)
    #X_pca = pca.fit_transform(X)
    #print(len(X_pca[0]))
    #y_pred = [0, 2, 1, 3]
    #y_test = [0, 1, 2, 3]
    #print(show_report("accuracy",y_test,y_pred))