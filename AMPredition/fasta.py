# count=0

input_file='seq.txt'
output_file='seq.fasta'
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    sequence_name = "sequence"  # 默认的序列名称
    for line in infile:
        line = line.strip()
        if line:
            outfile.write(f'>{sequence_name}\n{line}\n')






# with open('/tmp/pycharm_project_transformer/data/moses/elite_train.txt','r') as f:
#     with open('/tmp/pycharm_project_transformer/data/moses/elite_train.fasta','w') as f2:
#         for line in f:
#             f2.write('>'+str(count)+'\n'+line)
#             count+=1
#
# count1=0
# with open('/tmp/pycharm_project_transformer/data/moses/elite_test.txt','r') as f:
#     with open('/tmp/pycharm_project_transformer/data/moses/elite_test.fasta','w') as f2:
#         for line in f:
#             f2.write('>'+str(count1)+'\n'+line)
#             count1+=1
#
