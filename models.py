from preprocess import gen_features,create_X_and_y,merge_features,preprocess,preprocess_for_model
from model_selection import apply_best_estimator,show_report
#from model_selection import smote
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#import seaborn as sns
#import matplotlib as plt
import matplotlib
#matplotlib.use('GTK3Agg')  #I had to use GTKAgg for this to work, GTK threw errors
import matplotlib.pyplot as plt
#import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import joblib

'''
# Random Forest 
# Feature: AAC+PCP

clf = OneVsRestClassifier(estimator=RandomForestClassifier(class_weight='balanced',
                                                     max_depth=50,
                                                     max_features='sqrt',
                                                     n_estimators=150,
                                                     random_state=29))
'''
# SVM rbf 
# Feature: AAC

#clf = OneVsRestClassifier(estimator=SVC(C=100, class_weight='balanced', gamma=1,
#                                  random_state=29))

# SVM poly
# Feature: DPC 

#clf = OneVsRestClassifier(estimator=SVC(C=0.1, class_weight='balanced', degree=2,
#                                  gamma=0.01, kernel='poly', random_state=29))

'''
OneVsRestClassifier(estimator=SVC(C=0.1, class_weight='balanced', degree=2,
                                  gamma=0.01, kernel='poly', random_state=29))
DPC
              precision    recall  f1-score   support

           1       0.89      0.93      0.91       200
           2       0.89      0.83      0.86       138
           3       0.79      0.92      0.85        12

    accuracy                           0.89       350
   macro avg       0.86      0.89      0.87       350
weighted avg       0.89      0.89      0.89       350
'''
def choose_clf(algo):
    if algo == 'RF':
        clf = OneVsRestClassifier(estimator=RandomForestClassifier(
            max_depth=50, max_features='sqrt', n_estimators=150,random_state=29))
        feature = "AAC+PCP"
    elif algo == 'SVM_poly':
        clf = OneVsRestClassifier(estimator=SVC(C=0.1, degree=2,probability=True,
            gamma=0.01, kernel='poly', random_state=29))
        feature = "features/PAAC+features/DPC"
    elif algo == 'SVM_rbf':
        clf = OneVsRestClassifier(estimator=SVC(C=10, gamma=0.1,random_state=29,
            class_weight='balanced'))
        feature = "AAC"
    return clf,feature

def apply_model(feature,algo,clf):
    #print(clf)
    if "+" in feature:
        X,y = merge_features(feature,3)
    else:  
        X,y = create_X_and_y(feature + ".tsv",3)

    if "SVM" in algo:
        X_train,X_test,y_train,y_test = preprocess_for_model(X,y,"yes") #if scaled, then "yes" 
    elif "RF" in  algo:
        X_train,X_test,y_train,y_test = preprocess_for_model(X,y,"no")
    
    #fitting model
    clf_fitted = clf.fit(X_train,y_train)

    # saving model, uncomment to save  
    #joblib.dump(clf_fitted, "model.pkl")
    y_pred = apply_best_estimator(clf_fitted,X_test)
    cm = multilabel_confusion_matrix(y_test, y_pred)
    print(show_report("class_report",y_test,y_pred))

    return cm

def cv_scores(algo,clf):
    if "+" in feature:
        X,y = merge_features(feature,3)
    else:  
        X,y = create_X_and_y(feature + ".tsv",3)

    if "SVM" in algo:
        X_train,X_test,y_train,y_test = preprocess_for_model(X,y,"yes") #if scaled, then "yes" 
    elif "RF" in  algo:
        X_train,X_test,y_train,y_test = preprocess_for_model(X,y,"no")

    scores = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy')
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    return scores 


if __name__ == "__main__":
    algo = "SVM_poly"
    clf,feature = choose_clf(algo)
    print(feature)
    print(clf)
    #cm = apply_model(feature,algo,clf)
    #print(cv_scores(algo,clf))

    
    cm = apply_model(feature,algo,clf)
    #print(cm)
    #plt.plot([1, 2, 3])
    for i in range(0,3):
        #ax= plt.subplot()
        sns.set_context('poster')
        sns.heatmap(cm[i], annot=True,fmt='d')
        #ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
        plt.show()  
    