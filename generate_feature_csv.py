import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
from itertools import product

# Esse script é altamente ineficiente, otimizações virão depois quando as features forem realmente 
# selecionadas para o modelo

IUPAC_red = [
    'A', 'C', 'D', 'E', 'F', 
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y','X']

def dictWeightedSum(seq,d:dict):
    return sum([d[aa] for aa in seq])


def read_fasta(fp):
    ''' Reads .fasta file from _fp_ and returns a 
    generator of (name, sequence) tuples.
    '''
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                yield(name, ''.join(seq))
            name, seq = line, []
        else:
            seq.append(line)
    if name:
        yield (name, ''.join(seq))

def occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count+=1
        else:
            return count

def boman_index(seq):
    #based on: https://rdrr.io/cran/Peptides/src/R/boman.R
    scale = {
      'L' : 4.92, 'I' : 4.92,'V' : 4.04,
      'F' : 2.98, 'M' : 2.35,'W' : 2.33,
      'A' : 1.81, 'C' : 1.28,'T' : -2.57,
      'G' : 0.94, 'Y' : -0.14,'R' : -14.92,
      'S' : -3.40,'H' : -4.66,'Q' : -5.54,
      'K' : -5.55,'N' : -6.64,'E' : -6.81,
      'D' : -8.72,
    }

    return -1*sum([scale.get(aa,0) for aa in seq])/len(seq)

#Values from: https://github.com/dosorio/Peptides/blob/master/data/AAdata.RData

#https://rdrr.io/cran/Peptides/man/fasgaiVectors.html
#Ref:
# Liang, G., & Li, Z. (2007). Factor analysis scale of generalized amino acid
# information as the source of a new set of descriptors for elucidating the 
# structure and activity relationships of cationic antimicrobial peptides. 
# Molecular Informatics, 26(6), 754-763.

def FASGAI(seq,i:int):
  
    F1={
        'A':0.207, 
        'C':0.997, 
        'D':-1.298,
        'E':-1.349, 
        'F':1.247, 
        'G':-0.205, 
        'H':-0.27, 
        'I':1.524, 
        'K':-1387, 
        'L':1.2,
        'M':0.886, 
        'N':1.247, 
        'P':-0.407, 
        'Q':-0.88, 
        'R':-1.229,
        'S':-0.495, 
        'T':-0.032, 
        'V':-1.332, 
        'W':0.844, 
        'Y':0.329,}
    F2={
        "A": 0.821,
        "R": 0.378,
        "N": -0.939,
        "D": -0.444,
        "C": 0.021,
        "Q": 0.381,
        "E": 1.388,
        "G": -2.219,
        "H": 0.461,
        "I": 0.536,
        "L": 1.128,
        "K": 0.572,
        "M": 1.346,
        "F": 0.293,
        "P": -2.038,
        "S": -0.847,
        "T": -0.45,
        "W": -0.075,
        "Y": -0.858,
        "V": 0.545,}
    F3 = {
        "A": -1.009,
        "R": 0.516,
        "N": -0.428,
        "D": -0.584,
        "C": -1.419,
        "Q": -0.044,
        "E": -0.361,
        "G": -1.656,
        "H": -0.024,
        "I": 0.809,
        "L": 0.703,
        "K": 0.285,
        "M": 0.277,
        "F": 1.336,
        "P": -0.564,
        "S": -1.079,
        "T": -0.61,
        "W": 2.069,
        "Y": 1.753,
        "V": 0.029,
    }
    F4 = {
        "A": 1.387,
        "R": -0.328,
        "N": -0.397,
        "D": -0.175,
        "C": -2.08,
        "Q": -0.455,
        "E": 0.213,
        "G": 1.229,
        "H": -1.407,
        "I": 0.734,
        "L": 1.904,
        "K": 0.333,
        "M": -0.913,
        "F": -0.026,
        "P": -0.128,
        "S": 0.582,
        "T": 0.341,
        "W": -1.36,
        "Y": -0.479,
        "V": 1.026,
    }
    F5 = {
        "A": 0.063,
        "R": -0.052,
        "N": -0.539,
        "D": -0.259,
        "C": -0.799,
        "Q": -0.04,
        "E": 0.424,
        "G": -1.115,
        "H": 0.001,
        "I": -0.196,
        "L": 0.536,
        "K": -0.169,
        "M": 0.007,
        "F": 0.012,
        "P": 3.847,
        "S": 0.035,
        "T": 0.117,
        "W": -0.81,
        "Y": -0.835,
        "V": -0.229,
    }
    F6 = {
        "A": -0.6,
        "R": 2.728,
        "N": -0.605,
        "D": -1.762,
        "C": 0.502,
        "Q": 0.405,
        "E": -1.303,
        "G": -1.146,
        "H": 0.169,
        "I": 0.427,
        "L": -0.141,
        "K": 1.157,
        "M": -0.265,
        "F": -0.015,
        "P": -1.108,
        "S": -0.068,
        "T": 0.577,
        "W": -0.38,
        "Y": 0.289,
        "V": 1.038,
    }

    vectors = [F1,F2,F3,F4,F5,F6]

    return dictWeightedSum(seq,vectors[i])/len(seq)


#https://rdrr.io/cran/Peptides/man/crucianiProperties.html
#Ref:
# Cruciani, G., Baroni, M., Carosati, E., Clementi, M., Valigi, R., and Clementi, S.
#  (2004) Peptide studies by means of principal properties of amino acids derived 
# from MIF descriptors. J. Chemom. 18, 146-155.
def crucianiProps(seq,i):
    PP1 = {
        "A": -0.96,
        "R": 0.8,
        "N": 0.82,
        "D": 1,
        "C": -0.55,
        "E": 0.94,
        "Q": 0.78,
        "G": -0.88,
        "H": 0.67,
        "I": -0.94,
        "L": -0.9,
        "K": 0.6,
        "M": -0.82,
        "F": -0.85,
        "P": -0.81,
        "S": 0.41,
        "T": 0.4,
        "W": 0.06,
        "Y": 0.31,
        "V": -1,
    }
    PP2 = {
        "A": -0.76,
        "R": 0.63,
        "N": -0.57,
        "D": -0.89,
        "C": -0.47,
        "E": -0.54,
        "Q": -0.3,
        "G": -1,
        "H": -0.11,
        "I": -0.05,
        "L": 0.03,
        "K": 0.1,
        "M": 0.03,
        "F": 0.48,
        "P": -0.4,
        "S": -0.82,
        "T": -0.64,
        "W": 1,
        "Y": 0.42,
        "V": -0.43,
    }
    PP3 = {
        "A": 0.31,
        "R": 0.99,
        "N": 0.02,
        "D": -1,
        "C": 0.19,
        "E": -0.99,
        "Q": -0.38,
        "G": 0.49,
        "H": 0.37,
        "I": -0.18,
        "L": -0.24,
        "K": 1,
        "M": -0.08,
        "F": -0.58,
        "P": -0.07,
        "S": 0.57,
        "T": 0.37,
        "W": -0.47,
        "Y": -0.2,
        "V": -0.14,
    }

    vectors = [PP1,PP2,PP3]

    return dictWeightedSum(seq,vectors[i])/len(seq)

#https://rdrr.io/cran/Peptides/src/R/mswhimScores.R

def MSWHIM_scores(seq,i):
    vectors = [
        {
            "A": -0.73,
            "C": -0.66,
            "D": 0.11,
            "E": 0.24,
            "F": 0.76,
            "G": -0.31,
            "H": 0.84,
            "I": -0.91,
            "K": -0.51,
            "L": -0.74,
            "M": -0.7,
            "N": 0.14,
            "P": -0.43,
            "Q": 0.3,
            "R": -0.22,
            "S": -0.8,
            "T": -0.58,
            "V": -1,
            "W": 1,
            "Y": 0.97,
        },
        {
            "A": 0.2,
            "C": 0.26,
            "D": -1,
            "E": -0.39,
            "F": 0.85,
            "G": -0.28,
            "H": 0.67,
            "I": 0.83,
            "K": 0.08,
            "L": 0.72,
            "M": 1,
            "N": 0.2,
            "P": 0.73,
            "Q": 1,
            "R": 0.27,
            "S": 0.61,
            "T": 0.85,
            "V": 0.79,
            "W": 0.98,
            "Y": 0.66,
        },
        {
            "A": -0.62,
            "C": -0.27,
            "D": -0.96,
            "E": -0.04,
            "F": -0.34,
            "G": -0.75,
            "H": -0.78,
            "I": -0.25,
            "K": 0.6,
            "L": -0.16,
            "M": -0.32,
            "N": -0.66,
            "P": -0.6,
            "Q": -0.3,
            "R": 1,
            "S": -1,
            "T": -0.89,
            "V": -0.58,
            "W": -0.47,
            "Y": -0.16
        }
    ]

    return dictWeightedSum(seq,vectors[i])/len(seq)

# https://rdrr.io/cran/Peptides/man/protFP.html

def protFP(seq,i):
    vectors= [{
                "A":-0.1,
        "C":4.62,
        "D":-6.61,
        "E":-5.1,
        "F":6.76,
        "G":-5.7,
        "H":0.17,
        "I":6.58,
        "K":-4.99,
        "L":5.76,
        "M":5.11,
        "N":-4.88,
        "P":-3.82,
        "Q":-3.95,
        "R":-2.79,
        "S":-4.57,
        "T":-2,
        "V":5.04,
        "W":7.33,
        "Y":3.14,
            },
        {"A":-4.94,
        "C":-3.54,
        "D":0.94,
        "E":2.2,
        "F":0.88,
        "G":-8.72,
        "H":2.14,
        "I":-1.73,
        "K":5,
        "L":-1.33,
        "M":0.19,
        "N":0.81,
        "P":-2.31,
        "Q":2.88,
        "R":6.6,
        "S":-2.55,
        "T":-1.77,
        "V":-2.9,
        "W":4.55,
        "Y":3.59,},
        {
            "A":-2.13,
        "C":1.5,
        "D":-3.04,
        "E":-3.59,
        "F":0.89,
        "G":4.18,
        "H":1.2,
        "I":-2.49,
        "K":0.7,
        "L":-1.71,
        "M":-1.02,
        "N":0.14,
        "P":3.45,
        "Q":-0.83,
        "R":1.21,
        "S":-0.67,
        "T":-0.7,
        "V":-2.29,
        "W":2.77,
        "Y":2.45,

        },{
            "A":1.7,
        "C":-1.26,
        "D":-4.58,
        "E":-2.26,
        "F":-1.12,
        "G":-1.35,
        "H":0.71,
        "I":1.09,
        "K":3,
        "L":0.63,
        "M":0.15,
        "N":-0.14,
        "P":1,
        "Q":0.52,
        "R":2.07,
        "S":1.11,
        "T":1.02,
        "V":1.38,
        "W":-2.41,
        "Y":-1.27,
        }  ,
        {
            "A":-0.39,
        "C":3.27,
        "D":0.48,
        "E":-2.14,
        "F":-0.49,
        "G":-0.31,
        "H":1.16,
        "I":-0.34,
        "K":-1.23,
        "L":-1.7,
        "M":0.13,
        "N":1.23,
        "P":-3.22,
        "Q":0.9,
        "R":1.67,
        "S":0.99,
        "T":1.06,
        "V":0.06,
        "W":-1.08,
        "Y":-0.06,
        },
        {
            "A":1.06,
        "C":-0.34,
        "D":-1.31,
        "E":1.35,
        "F":-0.55,
        "G":2.91,
        "H":-0.38,
        "I":-0.28,
        "K":1.41,
        "L":0.71,
        "M":-0.3,
        "N":-0.65,
        "P":-3.54,
        "Q":0.55,
        "R":0.76,
        "S":-1.02,
        "T":-1.2,
        "V":0.08,
        "W":1.04,
        "Y":-0.29,
        },
        {
            "A":-1.39,
        "C":-0.47,
        "D":0.1,
        "E":-0.45,
        "F":-0.87,
        "G":0.32,
        "H":-1.85,
        "I":1.97,
        "K":0.19,
        "L":-0.05,
        "M":-2.95,
        "N":1.02,
        "P":-0.36,
        "Q":-0.08,
        "R":0,
        "S":0.11,
        "T":0.74,
        "V":1.79,
        "W":0.23,
        "Y":1.99,
        },
        {
            "A":0.97,
        "C":-0.23,
        "D":0.94,
        "E":-1.31,
        "F":1.05,
        "G":-0.11,
        "H":-2.79,
        "I":-0.92,
        "K":0.87,
        "L":-0.51,
        "M":0.5,
        "N":-1.94,
        "P":-0.3,
        "Q":0.64,
        "R":0.32,
        "S":0.65,
        "T":1.65,
        "V":-0.38,
        "W":0.59,
        "Y":0.3,
        }  ]

    return dictWeightedSum(seq,vectors[i])/len(seq)

#https://rdrr.io/cran/Peptides/src/R/stScales.R

#https://rdrr.io/cran/Peptides/src/R/tScales.R

#https://rdrr.io/cran/Peptides/man/vhseScales.html

#https://rdrr.io/cran/Peptides/man/zScales.html

if __name__ == "__main__":

    d = {'Sequence':[],'Label':[]}

    for name,seq in read_fasta(open('avps_labeledV3.cdhit.pep')):
        d['Sequence'].append(seq)
        d['Label'].append('Antiviral')
    #for name,seq in read_fasta(open('data/final_data/not_antiviral.fasta')):
    #    d['Sequence'].append(seq)
    #    d['Label'].append('Not Antiviral')

    df = pd.DataFrame(d)
    df['Size'] = df.Sequence.apply(len)

    df['Sequence_noX'] = df.Sequence.apply(lambda x: x.replace('X',''))
    #Some fisical properties

    #df['IP'] = df.Sequence.apply(lambda x : PA(x).isoelectric_point())
    df['MW'] = df.Sequence_noX.apply(lambda x : PA(x).molecular_weight())
    #df['Aromaticity'] = df.Sequence.apply(lambda x : PA(x).aromaticity())
    #df['Inst_index'] = df.Sequence_noX.apply(lambda x : PA(x).instability_index())
    #df['gravy'] = df.Sequence_noX.apply(lambda x : PA(x).gravy())
    
    #df['helix_frac'] = df.Sequence.apply(lambda x : PA(x).secondary_structure_fraction()[0])
    #df['turn_frac'] = df.Sequence.apply(lambda x : PA(x).secondary_structure_fraction()[1])
    #df['sheet_frac'] = df.Sequence.apply(lambda x : PA(x).secondary_structure_fraction()[2])

    #df['MEC_cysteines'] = df.Sequence.apply(lambda x : PA(x).molar_extinction_coefficient()[0])
    #df['MEC_cystines'] = df.Sequence.apply(lambda x : PA(x).molar_extinction_coefficient()[1])

    #df['boman_index'] = df.Sequence.apply(boman_index)
    # for i in range(5):
    #     df[f'flexibility_{i}'] = df.Sequence_noX.apply(lambda x : PA(x).flexibility()[i])

    df['charge_at_pH_5'] = df.Sequence.apply(lambda x : PA(x).charge_at_pH(5))
    df['charge_at_pH_7'] = df.Sequence.apply(lambda x : PA(x).charge_at_pH(7))
    df['charge_at_pH_9'] = df.Sequence.apply(lambda x : PA(x).charge_at_pH(9))

    for i in range(6):
        df[f'FASGAI_F{i+1}'] = df.Sequence_noX.apply(lambda x: FASGAI(x,i))

    for i in range(3):
        df[f'cruciani_PP{i+1}'] = df.Sequence_noX.apply(lambda x: crucianiProps(x,i))
    
    for i in range(3):
        df[f'MSWHIM{i+1}'] = df.Sequence_noX.apply(lambda x: MSWHIM_scores(x,i))
    
    for i in range(8):
        df[f'protFP{i+1}'] = df.Sequence_noX.apply(lambda x: protFP(x,i))
  
    # df['PseudoAAC'] = df.Sequence.apply(Pseudo_AAC)

    # Save count values
    for aa in IUPAC_red:
        df[f'count_{aa}'] = df.Sequence.apply(lambda x : x.count(aa))
    
    for aa in IUPAC_red:
        df[f'AAC_{aa}'] = df[f'count_{aa}']/df.Size
        
        
    for aa,aa2 in product(IUPAC_red,repeat=2):    
        if f'count_{aa2}{aa}' in df.columns:
            df[f'count_{aa2}{aa}'] = df.Sequence.apply(lambda x : occurrences(x,f'{aa}{aa2}'))
        else:
            df[f'count_{aa}{aa2}'] = df.Sequence.apply(lambda x : occurrences(x,f'{aa}{aa2}'))

    for aa,aa2 in product(IUPAC_red,repeat=2):
        if f'count_{aa}{aa2}' in df.columns:
            df[f'DCC_{aa}{aa2}'] = df[f'count_{aa}{aa2}']/df.Size
    
        
    #for aa,aa2,aa3 in product(IUPAC_red,repeat=3):
    #    if f'count_{aa3}{aa2}{aa}' in df.columns:
    #        df[f'count_{aa3}{aa2}{aa}'] = df.Sequence.apply(
    #            lambda x : occurrences(x,f'{aa}{aa2}{aa3}'))
    #    else:
    #        df[f'count_{aa}{aa2}{aa3}'] = df.Sequence.apply(
    #            lambda x : occurrences(x,f'{aa}{aa2}{aa3}'))

    #for aa,aa2,aa3 in product(IUPAC_red,repeat=3):
    #    if f'count_{aa}{aa2}{aa3}' in df.columns:
    #            df[f'TCC_{aa}{aa2}{aa3}'] = df[f'count_{aa}{aa2}{aa3}']/df.Size

    # df.drop()

    # Remove repetitions
  

    #Aliphatic index
    # based on https://rdrr.io/cran/Peptides/src/R/aindex.R
    #df['Aliphatic_index'] = df['AAC_A'] + 2.9*df['AAC_V'] + 3.9*(df['AAC_L']+df['AAC_I'])


    # df = df.drop(columns=['Sequence_noX']+[name for name in df.columns if 'count' in name])
    df = df.drop(columns=['Sequence_noX'])
    df.to_csv('features.csv',sep='\t',index=False)
    print(len(df.columns))