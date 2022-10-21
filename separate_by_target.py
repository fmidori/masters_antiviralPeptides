#import sys

# Create dictionary with target and label
def read_target_gen_label(tsv):
    dic = {}
    with open(tsv, 'r') as t:
        for line in t:
            target = line.split('\t')[0].strip()
            dic[target] = line.split('\t')[1].strip()
    t.close()
    return dic

# number labels
def number_label(label):
    num = "-"
    #if label == "Membrane":
    #    num = 1
    #elif label == "Replication":
    #   num = 2
    #elif label == "Assembly":
    #    num = 3
    if label == "D":
        num = 4
    #elif label == "Syncytium formation":
    #    num = 5
    return str(num)

# Generate fasta with AVP_numeber|number_of_label
def gen_fasta_with_label(tsv,dic):
    fasta = open("avps_labeledV3_only4.pep", "w")
    with open(tsv, 'r') as t:
        cont = 1
        for line in t:
            seq = line.split('\t')[1].strip()
            target = line.split('\t')[10].strip()
            if target in dic:
                label = number_label(str(dic[target]))
            else: label = '-'
            if label != '-':
                fasta.write(">AVP_" + str(cont) + "|" + str(label) + "\n" + str(seq) + '\n')
            cont += 1
    t.close()
    return None 


if __name__ == "__main__":
    dic = read_target_gen_label("original_data/target_to_labelV3.txt")
    gen_fasta_with_label("old_data/avps.tsv", dic)
