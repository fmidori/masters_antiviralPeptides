from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from preprocess import create_X_and_y,merge_features
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
matplotlib.use('GTK3Agg')  #I had to use GTKAgg for this to work, GTK threw errors
import matplotlib.pyplot as plt
import plotly.express as px

def apply_pca2D_for_plot(X,y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)


    df_target = pd.DataFrame(y, columns=['target'])
    principal_df = pd.DataFrame(data=principalComponents,
                    columns=['PCA1','PCA2'])
    final_df = pd.concat([principal_df, df_target[['target']]], axis=1)    
    #print(final_df.head())
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = [1,2,3]
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = final_df['target'] == target
        ax.scatter(final_df.loc[indicesToKeep, 'PCA1']
                , final_df.loc[indicesToKeep, 'PCA2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()     
    plt.show()  

def apply_pca3D_for_plot(X,y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    print(pca.explained_variance_ratio_)


    df_target = pd.DataFrame(y, columns=['target'])
    principal_df = pd.DataFrame(data=principalComponents,
                    columns=['PCA1','PCA2','PCA3'])
    final_df = pd.concat([principal_df, df_target[['target']]], axis=1)    
    #print(final_df.head())
    fig = px.scatter_3d(final_df, x='PCA1', y='PCA2', z='PCA3',
                    color='target',
                    title="3D Scatter Plot")
    fig.update_traces(legendgroup=3, selector=dict(type='scatter3d'))    
    fig.show()
    return None 

def create_X_and_y_for_analysis(tsv,num_of_labels):
    labels = [i+1 for i in range(int(num_of_labels))]
    with open(tsv, 'r') as t:
        X = []
        y = []
        for line in t:
            if not line.startswith('#'):
                label = int(line.split('\t')[0].split('|')[1])
                if label in labels:
                    y.append(label)
                    x = line.replace('\t',',').replace('\n','').split(',')
                    x.pop(0)
                    x = [float(i) for i in x]
                    X.append(x)
            else:
                columns = line.split('\t')
                columns.pop(0)
                #columns[0] = 'target'
        y = np.array(y)
    return X,y,columns

def merge_features_for_analysis(feature,num_of_labels):
    X = []
    y = []
    columns = []
    feature_list = feature.split("+")
    for feat in feature_list:
        X_to_merge,y,col = create_X_and_y_for_analysis(feat + ".tsv",num_of_labels)
        #print(X_to_merge[0])
        #col.pop(0)
        if not X:
            X = X_to_merge
            columns = col 
        else:
            for i in range(len(X)):
                X[i].extend(X_to_merge[i])
            print(len(X[0]))
            columns = columns + col 
    return X,y,columns

def best_in_feature(X,y,columns_features):
    df_target = pd.DataFrame(y, columns=['target'])
    principal_df = pd.DataFrame(data=X,
                    columns=columns_features)
    final_df = pd.concat([principal_df, df_target[['target']]], axis=1)
    #print(final_df.head())
    for i in range(1,4):
        print("\nclass:",i)
        df = final_df.groupby(['target']).mean().sort_values(by=i,ascending=False, axis=1)
        print(df.iloc[:,0:5])

    return None    

if __name__ == "__main__":
    feature = "features/PAAC"
    #+features/PAAC"
    #print(feature)
    if "+" in feature:
        X,y,columns = merge_features_for_analysis(feature,3)
    else:
        X,y,columns = create_X_and_y_for_analysis(feature + ".tsv",3)
    #best_in_feature(X,y,columns)
    #print(len(X[0]))
    #print(columns)
    #apply_pca3D_for_plot(X,y)

    df = pd.read_csv('feature_importance.tsv', sep='\t')
    fig = plt.figure()
    #sns.set_context('poster')
    ax = sns.barplot(x="feature_1", y="freq", hue="classe", data=df.loc[df['feature'] == 'DPC'])
    plt.xlabel("Dipeptide")
    plt.show()

    
