with open('/tmp/pycharm_project_pepGCT/prediction/seq2.txt','r') as f:
    with open('/tmp/pycharm_project_pepGCT/prediction/seq3.txt','w') as f2:
        for line in f:
            line=line.strip()
            f2.write(line + '\n')
            if len(line)>10:
                for i in range(len(line)-10+1):
                    seq=line[i:i+10]
                    f2.write(seq+'\n')



