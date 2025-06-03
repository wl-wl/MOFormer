import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from modlamp.descriptors import GlobalDescriptor
# 读取CSV文件并转化为DataFrame
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 尝试使用 SimHei 字体，失败时使用 Arial 字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 在Jupyter Notebook中使用以下代码


# file_path = '/tmp/pycharm_project_340/data/moses/prop_ten_train.csv'  # 请替换成你的CSV文件路径
# df = pd.read_csv(file_path)
# length_list, MW_list, Charge_list, ChargeDensity_list = [], [], [], []
# pI_list, InstabilityInd_list, Aromaticity_list = [], [], []
# AliphaticInd_list, BomanInd_list, HydrophRatio_list = [], [], []
# with open('/tmp/pycharm_project_pepGCT/prediction/seq2.txt','r') as f:
#     for line in f:
#         seq=line.strip()
#         desc_all = GlobalDescriptor(seq)
#         desc_all.calculate_all()
#         desc_all = desc_all.descriptor
#         desc_all = desc_all.ravel()
#
#         length_list.append(str(desc_all[0]))
#         MW_list.append(str(desc_all[1]))
#         Charge_list.append(str(desc_all[2]))
#         ChargeDensity_list.append(str(desc_all[3]))
#         pI_list.append(str(desc_all[4]))
#         InstabilityInd_list.append(str(desc_all[5]))
#         Aromaticity_list.append(str(desc_all[6]))
#         AliphaticInd_list.append(str(desc_all[7]))
#         BomanInd_list.append(str(desc_all[8]))
#         HydrophRatio_list.append(str(desc_all[9]))
#
#
# df2 = pd.DataFrame({'Column1': length_list, 'Column2': MW_list, 'Column3': Charge_list,\
#                   'Column4': ChargeDensity_list, 'Column5': pI_list, 'Column6': InstabilityInd_list, \
#                    'Column7': Aromaticity_list, 'Column8': AliphaticInd_list, 'Column9': BomanInd_list, \
#                   'Column10': HydrophRatio_list})
#
# # 画出每列的小提琴图
# plt.figure(figsize=(16, 8))
# for column in df.columns:
#     plt.subplot(2, 5, df.columns.get_loc(column) + 1)  # 创建一个2x5的子图，每列对应一个特征
#     sns.histplot(df[column], bins=30, kde=True)
#     plt.title(column)
#
# plt.tight_layout()  # 调整子图布局
# plt.show()







import seaborn as sns
import matplotlib.pyplot as plt
# import statsmodels.api as sm
# with open('/tmp/pycharm_project_pepGCT/p_mic0(mic_hemo_toxi).txt','r') as f:
#     for line1 in f:
#         seq, mic, hemo, toxi = line1.strip().split('\t')
#         mic, hemo, toxi=float(mic), float(hemo), float(toxi)
#         if -1<=mic<=3 and hemo<=0.8 and toxi<=0.8:
#             count+=1
# print(count)
# mic_list=[]
# mic_list2=[]
# with open('/tmp/pycharm_project_pepGCT/p_mic0(mic_hemo_toxi).txt','r') as f,open('/tmp/pycharm_project_pepGCT/mic_b.txt', 'r') as f0:
#     for line1, line2 in zip(f, f0):
#         mic, hemo, toxi = line1.strip().split("),")
#         seq, mic2 = line2.strip().split('\t')
#         mic_list.append(float(mic[9:15]))
#         mic_list2.append(float(mic2))
# print(mic,mic2)

# sns.kdeplot(mic_list, label='List 1', shade=True)
# sns.kdeplot(mic_list2, label='List 2', shade=True)
# plt.legend()
# plt.show()
# qqplot1 = sm.qqplot(mic_list, line='s')
# qqplot2 = sm.qqplot(mic_list2, line='s')
#
# plt.show()
# sum=0
# count=0
# with open('/tmp/pycharm_project_pepGCT/mic_b.txt','r') as f:
#     for line in f:
#         seq,mic=line.split('\t')
#         if float(mic)<0:
#             print(mic,len(seq))
#             count+=1
#             sum+=len(seq)
# print(sum/count,count)

# count=0
# with open('/tmp/pycharm_project_pepGCT/total_copy.txt','r') as f:
#     for line1 in f:
#         seq, mic, hemo, toxi = line1.strip().split('\t')
#         mic, hemo, toxi=float(mic), float(hemo), float(toxi)
#         if -1<=mic<=3 and hemo<=0.8 and toxi<=0.8:
#             count+=1
# print(count)
#965
#7010

# with open('/tmp/pycharm_project_pepGCT/p_mic0(mic_hemo_toxi).txt','r') as f,open('/tmp/pycharm_project_pepGCT/mic_b.txt', 'r') as f0:
#     with open('/tmp/pycharm_project_pepGCT/p_mic0(mic_hemo_toxi).csv', 'w') as f2:
#         for line1, line2 in zip(f, f0):
#             mic,hemo,toxi=line1.strip().split("),")
#             # mic=mic[9:15]
#             hemo=hemo[7:13]
#             toxi=toxi[7:13]
#             seq,mic=line2.strip().split('\t')
#             # f2.write(seq+'\t'+mic+'\t'+hemo+'\t'+toxi+'\n')
#             f2.write(mic + ',' + hemo + ',' + toxi + '\n')
#             # print(line2)



with open('/tmp/pycharm_project_pepGCT/pp1227_mic2(mic_toxi).txt','r') as f:
    with open('/tmp/pycharm_project_pepGCT/pp1227_mic2(mic_toxi)_copy.txt', 'w') as f2:
        for line in f:
            _, _, pre2,_ = line.split(',')
            strs=line[9]
            for i in line[10:]:
                if i==']':
                    break
                strs+=i
            f2.write(strs+','+str(pre2[7:13]+'\n'))


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial import ConvexHull
#
# # 生成示例数据
# num_points = 10
# objectives = np.random.rand(num_points, 2)
#
# print(objectives)
#
# # 计算Pareto前沿
# def pareto_frontier(points):
#     points = sorted(points, key=lambda x: x[0])
#     frontier = [points[0]]
#
#     for i in range(1, len(points)):
#         if points[i][1] < frontier[-1][1]:
#             frontier.append(points[i])
#
#     return np.array(frontier)
#
#
# pareto_points = pareto_frontier(objectives)
#
#
# # 找到Pareto前沿的关键点（kneepoints）
# def find_kneepoints(points):
#     hull = ConvexHull(points)
#     kneepoints = []
#
#     for simplex in hull.simplices:
#         kneepoints.extend(simplex.tolist())
#
#     return list(set(kneepoints))
#
#
# kneepoints_indices = find_kneepoints(pareto_points)
# kneepoints = pareto_points[kneepoints_indices]
#
# # 可视化Pareto前沿线条和关键点
# plt.scatter(pareto_points[:, 0], pareto_points[:, 1], label='Pareto Frontier',color='black')
# plt.scatter(kneepoints[:, 0], kneepoints[:, 1], color='red', s=80,marker='o', label='Kneepoints')
# plt.scatter(kneepoints[0, 0], kneepoints[0, 1], color='orange', marker='o', s=80,label='Kneepoints')
# plt.scatter(kneepoints[-1, 0], kneepoints[-1, 1], color='orange', marker='o', s=80,label='Kneepoints')
# plt.xlabel('Objective 1')
# plt.ylabel('Objective 2')
# plt.rcParams['savefig.dpi'] = 300
#
# plt.title('Pareto Frontier with Kneepoints')
# plt.savefig('figure1_knee_point.png')
# plt.legend()
# # plt.grid(True)
# plt.show()
#
#
