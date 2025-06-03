import numpy as np
def CTD():
    protein_seq_dict1 = {}
    protein_index1 = 1
    with open('/tmp/pycharm_project_763/CTD.txt', 'r') as t:
        for line in t:
            seq = line[:-1].split('\t')[1:]
            protein_seq_dict1[protein_index1] = seq
            protein_index1= protein_index1 + 1
    print(len(protein_seq_dict1[1]))

    for j in range(1,741):
        for i in range(39):
            protein_seq_dict1[j][i]=protein_seq_dict1[j][i][:5]
            # print(protein_seq_dict1[j][i][:4])
    # print((protein_seq_dict1))
    fe = []
    for j in range(1, 741):
        fe.append(protein_seq_dict1[j])
    fe = np.array(fe)
    return fe
CTD()
def fe():  #aac
    protein_seq_dict1 = {}
    protein_index1 = 1
    with open('/tmp/pycharm_project_763/AAC.txt', 'r') as t:
        for line in t:
            seq = line[:-1].split('\t')[1:]
            protein_seq_dict1[protein_index1] = seq
            protein_index1= protein_index1 + 1
    # print(protein_seq_dict1)

    for j in range(1,741):
        for i in range(20):
            protein_seq_dict1[j][i]=protein_seq_dict1[j][i][:5]
            # print(protein_seq_dict1[j][i][:4])
    # print((protein_seq_dict1))
    fe = []
    for j in range(1, 741):
        fe.append(protein_seq_dict1[j])
    fe = np.array(fe)
    return fe
def ac():
    label = []
    protein_seq_dict = {}
    protein_index = 1
    with open('/tmp/pycharm_project_763/AC.txt', 'r') as fp:
        for line in fp:
            if line[0] == '>':
                values = line[1:].strip('').split('|')
                label_temp = values[1]
                protein = values[0]
                if label_temp == '1\n':
                    label.append(1)
                else:
                    label.append(0)
            else:
                seq = line[:-1].split('\t')
                protein_seq_dict[protein_index] = seq
                #             print(seq)
                protein_index = protein_index + 1
    # print(protein_seq_dict)
    lh = []
    for j in range(1, 741):
        for k in range(0,42):
            protein_seq_dict[j][k]=float(protein_seq_dict[j][k])
        lh.append(protein_seq_dict[j])
        # print(protein_seq_dict[1])
    lh = np.array(lh)
    return lh,label

import scipy.stats.stats as st
import pandas as pd
def AAcal(seqcont):
    v=[]
    for i in range(len(seqcont)):
        vtar=seqcont[i]
        vtarv=[]
        vtar7=0
        vtar8=0
        vtar9=0
        s = pd.Series(vtar)
        vtar3=np.mean(vtar)  # These 4 dimensions are relevant statistical terms
        vtar4=st.kurtosis(vtar)
        vtar5=np.var(vtar)
        vtar6=st.skew(vtar)
        #for p in range(len(vtar)): # These 3 dimensions are inspired by PAFIG algorithm
            #vtar7=vtar[p]**2+vtar7
            #if vtar[p]>va:
                #vtar8=vtar[p]**2+vtar8
            #else:
                #vtar9=vtar[p]**2+vtar9
        vcf1=[]
        vcf2=[]
        for j in range(len(vtar)-1): #Sequence-order-correlation terms
            vcf1.append((vtar[j]-vtar[j+1]))
        for k in range(len(vtar)-2):
            vcf2.append((vtar[k]-vtar[k+2]))
        vtar10=np.mean(vcf1)
        vtar11=np.var(vcf1)
        vtar11A=st.kurtosis(vcf1)
        vtar11B=st.skew(vcf1)
        vtar12=np.mean(vcf2)
        vtar13=np.var(vcf2)
        vtar13A=st.kurtosis(vcf2)
        vtar13B=st.skew(vcf2)
        vtarv.append(vtar3)
        vtarv.append(vtar4)
        vtarv.append(vtar5)
        vtarv.append(vtar6)
        #vtarv.append(vtar7/len(vtar))
        #vtarv.append(vtar8/len(vtar))
        #vtarv.append(vtar9/len(vtar))
        vtarv.append(vtar10)
        vtarv.append(vtar11)
        vtarv.append(vtar11A)
        vtarv.append(vtar11B)
        vtarv.append(vtar12)
        vtarv.append(vtar13)
        vtarv.append(vtar13A)
        vtarv.append(vtar13B)
        v.append(vtarv)
    return v
ac,label=ac()
def deal():
    ac_p=AAcal(ac)
    return ac_p,label
def gaac():#gaac
    protein_seq_dict1 = {}
    protein_index1 = 1
    with open('/tmp/pycharm_project_763/GAAC.txt', 'r') as t:
        for line in t:
            seq = line[:-1].split('\t')[1:]
            protein_seq_dict1[protein_index1] = seq
            protein_index1= protein_index1 + 1
    # print(protein_seq_dict1)

    for j in range(1,741):
        for i in range(5):
            protein_seq_dict1[j][i]=protein_seq_dict1[j][i][:10]
            # print(protein_seq_dict1[j][i][:4])
    # print((protein_seq_dict1))
    fe = []
    for j in range(1, 741):
        fe.append(protein_seq_dict1[j])
    fe = np.array(fe)
    return fe