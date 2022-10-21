import joblib
import argparse
import subprocess
from iFeature import PAAC,DPC,readFasta
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from preprocess import merge_features,preprocess_for_model
#from Bio import SeqIO

parser = argparse.ArgumentParser(prog="predict.py",description= 'smt')
parser.add_argument('-fasta', metavar='<fasta file>' ,help='Specify fasta file')
parser.add_argument('-out', metavar='<name of file>' ,help='Specify output file name',default='results.tsv')

args = parser.parse_args()

clf = joblib.load("model.pkl")
scaler = joblib.load("std_scaler.pkl") 

def gen_feature_for_fasta(fasta):
    fastas = readFasta.readFasta(str(fasta))
    lambdaValue = 6 
    
    encodings = PAAC.PAAC(fastas, lambdaValue)
    dpc_enc = DPC.DPC(fastas)
    for j in dpc_enc:
        del j[0]

    for i in range(len(encodings)):
                encodings[i].extend(dpc_enc[i])
    encodings[0][0] = "peptide_id"
    #print(dpc_enc)
    return encodings

def predict_AVP(encodings,clf,scaler): 
    out = open(str(args.out),"w")
    peptides = []
    peptides_id = []
    for i in range(1,len(encodings)):
        peptide = encodings[i]
        id_peptide = peptide[0]
        peptides_id.append(str(id_peptide))
        peptide.pop(0)
        #print(len(peptide))
        pep_scaled = scaler.transform([peptide])
        peptides.append(pep_scaled[0])
    pred = clf.predict(peptides)
    #print(pred)
    prob = clf.predict_proba(peptides)
    proba = []
    for i in range(len(prob)):
        proba.append([0,0,0])
        for j in range(len(prob[i])):
            proba[i][j] = prob[i][j] * 100
        #print(id_peptide, pred)
    #print(proba)
    #print(peptides_id)

    for i in range(len(peptides_id)):
        out.write(peptides_id[i] + "\t" + str(proba[i][0]) + "\t" + str(proba[i][1]) + "\t" + str(proba[i][2]) + "\t" + str(pred[i]) + "\n" )
    return None 

#clf = joblib.load("model.pkl")
#print(clf)

if __name__ == "__main__":
    '''
    X,y = merge_features("features/PAAC+features/DPC",3)
    scaler = StandardScaler().fit(X)
    joblib.dump(scaler, 'std_scaler.pkl', compress=True)
    
    '''
    #print(len(X[0]))
    encodings = gen_feature_for_fasta(str(args.fasta))
    predict_AVP(encodings,clf,scaler)

    
    with open('results_DBAASP.tsv','r') as f:
        correct = 0
        wrong = 0 
        for line in f: 
            real = int(line.split('\t')[0].split('|')[1])
            pred = int(line.split('\t')[4])
            if real == pred:
                correct += 1
            elif (real == 4 ) and pred == 1:
                correct += 1
            elif (real == 4 ) and pred == 2:
                correct += 1
            else: wrong += 1 
    print(correct,wrong)
    