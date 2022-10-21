Scripts used for preprocessing data from antiviral peptides database, 
preparing data for machine learning models and application of ML to train and test. 

### Main scripts

1 - separate_by_target.py: Create labels from targets and generate fasta file with labeled AVPs data.

2 - preprocess.py: Generate features from iFeature package, create X and y, preprocess data with PCA, SMOTE and Scaler. 

3 - model_selection.py: Select best machine learning model with GridSearchCV. 


4 - predict.py: main script to predict, with best model chosen with prevous script.

5 - feature_analysis.py: Graphic analysis of features and PCA 2d and 3d.

## Other scripts: 
- new_seqs_from_db.py: Add new sequences to database 
- create_neg_dataset.py: Used to create negative database   
- generate_feature_csv.py|other_features.py:  Generate features with functions from iFeature
- models.py: Test models and saving to pkl file