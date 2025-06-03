from Bio.PDB import PDBParser
from Bio.PDB.DSSP import make_dssp_dict
import numpy as np
def read_dssp(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    dssp_data = []
    start = False
    for line in lines:
        if line.startswith("  #  RESIDUE AA STRUCTURE BP1 BP2"):
            start = True
            continue
        if start:
            fields = line.split()
            if len(fields) > 13:
                residue = fields[1]
                aa = fields[2]
                structure = fields[3]
                acc = fields[9]
                phi = fields[10]
                psi = fields[11]
                dssp_data.append((residue, aa, structure, acc, phi, psi))
    return dssp_data

# 读取 DSSP 文件
dssp_file = "/tmp/pycharm_project_763/feature/feature_test/dssp/1.dssp"
dssp_features = read_dssp(dssp_file)

# 打印提取的特征
for feature in dssp_features:
    print(feature)
