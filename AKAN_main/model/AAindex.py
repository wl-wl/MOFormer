






# #!/usr/bin/env python
# #_*_coding:utf-8_*_
#
# import sys, os, re, platform
# pPath = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(pPath)
# # import checkFasta
#
# def AAINDEX(fastas, **kw):
# 	# if checkFasta.checkFasta(fastas) == False:
# 	# 	print('Error: for "AAINDEX" encoding, the input fasta sequences should be with equal length. \n\n')
# 	# 	return 0
#
# 	AA = 'ARNDCQEGHILKMFPSTWYV'
#
# 	# fileAAindex = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\data2\AAindex.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data2/AAindex.txt'
# 	with open("/tmp/pycharm_project_763/data2/AAindex.txt") as f:
# 		records = f.readlines()[1:]
#
# 	AAindex = []
# 	AAindexName = []
# 	for i in records:
# 		AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
# 		AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)
#
# 	index = {}
# 	for i in range(len(AA)):
# 		index[AA[i]] = i
#
# 	encodings = []
# 	header = ['#']
# 	for pos in range(1, len(fastas[0][1]) + 1):
# 		for idName in AAindexName:
# 			header.append('SeqPos.' + str(pos) + '.' + idName)
# 	encodings.append(header)
#
# 	for i in fastas:
# 		name, sequence = i[0], i[1]
# 		code = [name]
# 		for aa in sequence:
# 			if aa == '-':
# 				for j in AAindex:
# 					code.append(0)
# 				continue
# 			for j in AAindex:
# 				code.append(j[index[aa]])
# 		encodings.append(code)
#
# 	return encodings
#
#
# def read_fasta(file_path):
#     """
#     读取fasta文件，并将其解析为一个包含序列名称和序列的列表。
#     每个元素是一个元组 (sequence_name, sequence)。
#     """
#     sequences = []
#     with open(file_path, 'r') as file:
#         sequence_name = ""
#         sequence = ""
#         for line in file:
#             line = line.strip()
#             if line.startswith(">"):
#                 if sequence_name and sequence:
#                     sequences.append([sequence_name, sequence])
#                 sequence_name = line[1:]  # 提取序列名称中的数字部分
#                 sequence = ""
#             else:
#                 sequence += line
#         if sequence_name and sequence:
#             sequences.append(([sequence_name, sequence]))
#     return sequences
#
#
# fastas = read_fasta("/tmp/pycharm_project_763/data2/trainCPP.fasta")
# AA=AAINDEX(fastas)
# print(len(AA[0]))
# print(len(AA[0][0]))