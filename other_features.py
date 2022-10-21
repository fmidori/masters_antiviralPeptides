#from mylib.constants import IUPAC_red
#from mylib.util import partialDictRoundedSum

IUPAC_red = [
    'A', 'C', 'D', 'E', 'F', 
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'X', 'Y']

def partialDictRoundedSum(d, keys):
    return round(sum([d[k] for k in keys]), 3)


def AAC(seq):
    return [seq.count(aa)/len(seq) for aa in IUPAC_red]


def AntivppFeatures(seq):
    seq_size = len(seq)

    kyte_doolittle = {'A': 1.80, 'C': 2.50, 'D': -3.50, 'E': -3.50, 'F': 2.80,
                      'G': -0.40, 'H': -3.20, 'I': 4.50, 'K': -3.90, 'L': 3.80,
                      'M': 1.90, 'N': -3.50, 'P': -1.60, 'Q': -3.50, 'R': -4.50,
                      'S': -0.80, 'T': -0.70, 'V': 4.20, 'W': -0.90, 'Y': -1.30}

    molecular_weigth = {'A': 89.09, 'C': 121.15, 'D': 133.10, 'E': 147.13, 'F': 165.19,
                        'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
                        'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
                        'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.24, 'Y': 181.19}

    net_charge = {'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
                  'G': 0, 'H': 0, 'I': 0, 'K': 1, 'L': 0,
                  'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
                  'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0}

    net_hydrogen = {'A': 0, 'C': 0, 'D': 1, 'E': 1, 'F': 0,
                    'G': 0, 'H': 1, 'I': 0, 'K': 2, 'L': 0,
                    'M': 0, 'N': 2, 'P': 0, 'Q': 2, 'R': 4,
                    'S': 1, 'T': 1, 'V': 0, 'W': 1, 'Y': 1}

    aa_list = [
    'A', 'C', 'D', 'E', 'F', 
    'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'V', 'W', 'Y']

    aa_counts = {k: seq.count(k) for k in aa_list}

    aa_perc = {k: round(aa_counts[k]/seq_size, 3) for k in aa_list}

    # PROPERTIES Q-P

    aliphatic = partialDictRoundedSum(aa_perc, ['I', 'V', 'L'])

    negative_charged = partialDictRoundedSum(aa_perc, ['D', 'E'])

    total_charged = partialDictRoundedSum(aa_perc, ['D', 'E', 'K', 'H', 'R'])

    aromatic = partialDictRoundedSum(aa_perc, ['F', 'H', 'W', 'Y'])

    polar = partialDictRoundedSum(aa_perc, ['D', 'E', 'R', 'K', 'Q', 'N'])

    neutral = partialDictRoundedSum(aa_perc,
                                    ['A', 'G', 'H', 'P', 'S', 'T', 'Y'])

    hydrophobic = partialDictRoundedSum(aa_perc,
                                        ['C', 'F', 'I', 'L', 'M', 'V', 'W'])

    positive_charged = partialDictRoundedSum(aa_perc, ['K', 'R', 'H'])

    tiny = partialDictRoundedSum(aa_perc, ['A', 'C', 'D', 'G', 'S', 'T'])

    small = partialDictRoundedSum(aa_perc,
                                  ['E', 'H', 'I', 'L', 'K', 'M', 'N', 'P', 'Q', 'V'])

    large = partialDictRoundedSum(aa_perc, ['F', 'R', 'W', 'Y'])

    # SCALES

    kyleD = round(
        sum(
            [aa_counts[k]*kyte_doolittle[k] for k in aa_list]
        )/seq_size, 3)

    molW = round(sum([aa_counts[k]*molecular_weigth[k] for k in aa_list]), 3)

    netCharge = sum([aa_counts[k]*net_charge[k] for k in aa_list])

    netH = round(sum([aa_counts[k]*net_hydrogen[k] for k in aa_list]), 3)

    results =  [netH, netCharge, molW, kyleD] 
    results += [v for v in aa_perc.values()]
    results += [tiny, small, large, aliphatic, aromatic]
    results += [total_charged, negative_charged, positive_charged]
    results += [polar, neutral, hydrophobic]
    return results

def main(fasta):
    out = open("PCP.tsv","w")
    with open(fasta, 'r') as f:
        for line in f:
            if line.startswith('>'):
                peptide = line.split('>')[1].strip()
            else:
                seq = str(line.strip())
                #feature_vector = AntivppFeatures(seq)
                feature_vector = "\t".join([str(i) for i in AntivppFeatures(seq)])
                out.write(str(peptide) + "\t" + feature_vector + "\n")
    return None 


if __name__ == "__main__":
    main("avps_labeledV3.cdhit.pep")