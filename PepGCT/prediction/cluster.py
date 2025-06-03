from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 假设有一些蛋白质序列的列表
protein_sequences=[]
with open('/tmp/pycharm_project_pepGCT/prediction/seq2.txt') as f:
    for line in f:
        # seq, mic, hemo, toxi = line.strip().split('\t')
        seq=line.strip()
        protein_sequences.append(seq)
# 将所有氨基酸取并集，构建氨基酸字典
all_amino_acids = set("".join(protein_sequences))
amino_acid_dict = {amino_acid: i for i, amino_acid in enumerate(all_amino_acids)}

# 将蛋白质序列转换为特征向量
def sequence_to_vector(sequence):
    vector = np.zeros(len(all_amino_acids))
    for amino_acid in sequence:
        vector[amino_acid_dict[amino_acid]] += 1
    return vector

# 将所有蛋白质序列转换为特征矩阵
feature_matrix = np.array([sequence_to_vector(seq) for seq in protein_sequences])

# 创建 PCA 模型，选择要保留的主成分数量
n_components = 2
pca = PCA(n_components=n_components)

# 对特征矩阵进行 PCA 降维
reduced_data = pca.fit_transform(feature_matrix)

# 打印降维后的数据
print("原始数据形状:", feature_matrix.shape)
print("降维后的数据形状:", reduced_data.shape)
print("降维后的数据:")
print(reduced_data)


kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(reduced_data)

# 绘制聚类可视化
plt.figure(figsize=(8, 6))

for i in range(len(cluster_labels)):
    color = 'red' if cluster_labels[i] == 0 else 'blue'
    plt.scatter(reduced_data[i, 0], reduced_data[i, 1], c=color, marker='o', s=50)

# 设置图表标题和坐标轴标签
plt.title('PCA降维后的蛋白质序列聚类可视化')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# 显示图例
plt.legend(['Cluster 1', 'Cluster 2'])

# 显示图表
plt.show()

# 示例用法
