
def read_seqs(tsv):
    seqs = []
    with open(tsv, 'r') as t:
        for line in t:
            seq = str(line.split('\t')[0].strip())
            seqs.append(seq)
    t.close()
    return seqs

def missing_seqs(tsv,seqs):
    out = open('missing_seqs_avpdb.tsv','w')
    exc = ['B','O','Z','-','X']
    with open(tsv, 'r') as t:
        for line in t:
            new_seq = str(line.split(' ')[0].strip()).upper() # for dbaasp use 4 spaces 
            print(new_seq)
            if new_seq not in seqs:
                write = True
                for exception in exc:
                    if str(exception) in new_seq:
                        write = False
                if write == True:
                    out.write(new_seq + '\n')
    return None 

if __name__ == "__main__":
    seqs = read_seqs('antiviral_AMPs_seq.tsv')
    #print(seqs)
    missing_seqs('AVPDB_AVPs.tsv',seqs)