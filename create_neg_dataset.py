from Bio import SeqIO
import random 

def get_length_fasta(fasta):
    all_lengths = {}
    with open(fasta) as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            if len(record.seq) in all_lengths:
                all_lengths[len(record.seq)] += 1 
            else: all_lengths[len(record.seq)] = 1
    all_lengths_sorted = dict(sorted(all_lengths.items()))
    return all_lengths_sorted


def get_pep_with_length(fasta,dic):

    with open(fasta) as handle:
        len_dic = len(dic)
        i = 0  
        get_length = int(list(dic.keys())[i]) #first length to get
        times = int(list(dic.values())[i]) # times it appears 
        seqs_to_shuf = []
        seqs = []
        for record in SeqIO.parse(handle, 'fasta'): 
            length = int(len(record.seq))
            if length == get_length:
                seqs_to_shuf.append(record)
            elif (length != get_length) and (len(seqs_to_shuf)>0):
                random.shuffle(seqs_to_shuf)
                for cont in range(0,times):
                    seqs.append(seqs_to_shuf[cont])
                seqs_to_shuf = []
                i += 1 
                if i < len_dic:
                    get_length = int(list(dic.keys())[i]) 
                    times = int(list(dic.values())[i])
                else: 
                    break
    SeqIO.write(seqs, "nonAVPs.pep", "fasta")
    return None

            
if __name__ == "__main__":
    dic = get_length_fasta("avps_labeled.cdhit.pep")
    #dic = get_length_fasta("nonAVPs.pep")

    print(dic)
    #dic = {5:2,9:1,30:2}
    #get_pep_with_length("nonAMP_peptides.named.sorted.fasta",dic)
