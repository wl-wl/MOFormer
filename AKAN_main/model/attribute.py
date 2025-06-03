from Bio.PDB import PDBParser, DSSP

# 设置DSSP可执行文件路径
dssp_executable = "/home/wangli/anaconda3/envs/transformer-env/bin/mkdssp"

# 解析PDB文件
pdb_filename = "/tmp/pycharm_project_763/PDB/test/1.pdb"  # 替换为你的PDB文件路径
parser = PDBParser()
structure = parser.get_structure("protein", pdb_filename)

# 选择模型和链
model = structure[0]  # 使用第一个模型
chain = model['A']  # 使用链A，替换为你感兴趣的链

# 计算DSSP
dssp = DSSP(model, pdb_filename, dssp=dssp_executable)

# 打印DSSP结果
# for residue in chain:
#     res_id = residue.get_id()
#     if res_id in dssp:
#         dssp_key = (res_id[1], res_id[2].strip())
#         dssp_data = dssp[dssp_key]
#         print(f"Residue {res_id[1]}: {dssp_data}")

for key in dssp.keys():
    residue_index = key[1]  # 获取残基索引
    dssp_data = dssp[key]
    print(f"Residue {residue_index}: {dssp_data}")