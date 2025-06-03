import numpy as np
from collections import Counter
import pandas as pd
import scipy.stats.stats as st
def read_fasta(file_path):

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # 提取标签
                label = line.split('|')[1]
                label_list.append(label)
            else:
                # 提取蛋白质序列
                protein_list.append(line)

    return protein_list, label_list

protein_list = []
label_list = []

# 示例文件路径
file_path = '/tmp/pycharm_project_763/raw/QSP.txt'

# 调用函数读取文件
protein_list, label_list = read_fasta(file_path)

# 输出结果
# print("Protein Sequences:", protein_list)
# print("Labels:", label_list)


def calculate_aac(protein_list):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'

    # 初始化存储结果的列表
    aac_list = []

    for protein_sequence in protein_list:
        # 初始化AAC特征字典
        aac = {aa: 0 for aa in amino_acids}

        # 计算每种氨基酸在序列中的频率
        for aa in protein_sequence:
            if aa in aac:
                aac[aa] += 1

        # 将频率转换为比例
        sequence_length = len(protein_sequence)
        aac = [count / sequence_length for aa, count in aac.items()]

        # 将结果添加到列表中
        aac_list.append(aac)

    return aac_list

aac_features = calculate_aac(protein_list)


# def one_hot(protein_list):
#     # 定义所有20种氨基酸
#     amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
#
#     # 创建一个字典，将氨基酸映射到索引
#     aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
#
#     # 计算最长的蛋白质序列长度
#     max_length = max(len(protein) for protein in protein_list)
#
#     # 初始化存储结果的数组
#     one_hot_encoded_list = []
#
#     for protein_sequence in protein_list:
#         # 初始化一个矩阵来存储one-hot编码，每行代表一个氨基酸
#         one_hot_matrix = np.zeros((max_length, len(amino_acids)))
#
#         for i, aa in enumerate(protein_sequence):
#             if aa in aa_to_index:
#                 one_hot_matrix[i, aa_to_index[aa]] = 1
#
#         # 将矩阵添加到结果列表中
#         one_hot_encoded_list.append(one_hot_matrix)
#
#     return one_hot_encoded_list
#
# one_hot_encoded_list = one_hot(protein_list)

"""
(changdu-G+1)*window
"""
def EGAAC(sequences, window=5):
    if window < 1:
        print('Error: the sliding window should be greater than zero' + '\n\n')
        return None

    group = {
        'alphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'positive_charge': 'KRH',
        'negative_charge': 'DE',
        'uncharged': 'STCPNQ'
    }

    groupKeys = group.keys()

    encodings = []
    header = ['#']
    max_len = max(len(seq) for seq in sequences)  # Find the maximum length of sequences
    for w in range(1, max_len - window + 2):
        for g in groupKeys:
            header.append('SW.' + str(w) + '.' + g)
    # encodings.append(header)

    for sequence in sequences:
        code = []
        for j in range(len(sequence) - window + 1):
            subseq = sequence[j:j + window]
            count = Counter(subseq)
            myDict = {}
            for key in groupKeys:
                myDict[key] = sum(count[aa] for aa in group[key] if aa in count)
            for key in groupKeys:
                code.append(myDict[key] / window)
        encodings.append(code)

    return encodings

# 示例蛋白质序列列表

EGAAC_result = EGAAC(protein_list, window=5)
max_length = max(len(item) for item in EGAAC_result)

# 进行零填充
padded_encodings = np.array([np.pad(item, (0, max_length - len(item)), 'constant', constant_values=(0)) for item in EGAAC_result])
print(padded_encodings)

def BLOSUM62(sequences):
    # BLOSUM62 substitution matrix as a dictionary
    blosum62 = {
        'A': [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],  # A
        'R': [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],  # R
        'N': [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],  # N
        'D': [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],  # D
        'C': [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],  # C
        'Q': [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],  # Q
        'E': [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],  # E
        'G': [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],  # G
        'H': [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],  # H
        'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],  # I
        'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],  # L
        'K': [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],  # K
        'M': [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],  # M
        'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],  # F
        'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],  # P
        'S': [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],  # S
        'T': [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],  # T
        'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],  # W
        'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],  # Y
        'V': [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4],  # V
        '-': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # -
    }

    # Initialize the result list with header
    header = ['#']
    max_length = max(len(seq) for seq in sequences)
    for i in range(1, max_length * 20 + 1):
        header.append('blosum62.F' + str(i))
    encodings = []

    # Process each sequence
    for sequence in sequences:
        code = []
        for aa in sequence:
            code += blosum62.get(aa, blosum62['-'])  # Use gap scores if aa not found
        encodings.append(code)

    return encodings


BLOSUM62_result = BLOSUM62(protein_list)
print(BLOSUM62_result)

max_length = max(len(item) for item in BLOSUM62_result)

# 进行零填充 64*20
padded_encodings = np.array([np.pad(item, (0, max_length - len(item)), 'constant', constant_values=(0)) for item in BLOSUM62_result])
print(padded_encodings)

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
AAC_2=AAcal(aac_features)

print(AAC_2)