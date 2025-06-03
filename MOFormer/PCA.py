from Bio.SeqUtils import ProtParam
from Bio.Seq import Seq
import pandas as pd
from modlamp.descriptors import GlobalDescriptor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt




# train_data=[]
df1 = pd.read_csv('descriptors.tsv', delimiter='\t', header=None)
df2 = pd.read_csv('descriptors2.tsv', delimiter='\t', header=None)
# df3 = pd.read_csv('descriptors.tsv', delimiter='\t', header=None)

df1=df1.iloc[1:, 1:]
# print(df1.isnull().sum(),df1.shape)
df2=df2.iloc[1:, 1:]
# df3=df1.iloc[1:, 1:]

# train_data = pd.concat([df1, df2], axis=1)
train_data=df1
test_data=df2

# train_data = df.iloc[1:, 1:]
scaler = StandardScaler()
scaled_data_train = scaler.fit_transform(train_data)


pca = PCA(n_components=3)
pca_result_train = pca.fit_transform(scaled_data_train)
explained_var_train = pca.explained_variance_ratio_
print("Train explained variance ratio (PCA1–3):", explained_var_train)
print("Total explained variance (train):", np.sum(explained_var_train))


feature_names = [
    "A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y",
    "hydrophobicity_PRAM900101.G1","hydrophobicity_PRAM900101.G2","hydrophobicity_PRAM900101.G3",
    "hydrophobicity_ARGP820101.G1","hydrophobicity_ARGP820101.G2","hydrophobicity_ARGP820101.G3",
    "hydrophobicity_ZIMJ680101.G1","hydrophobicity_ZIMJ680101.G2","hydrophobicity_ZIMJ680101.G3",
    "hydrophobicity_PONP930101.G1","hydrophobicity_PONP930101.G2","hydrophobicity_PONP930101.G3",
    "hydrophobicity_CASG920101.G1","hydrophobicity_CASG920101.G2","hydrophobicity_CASG920101.G3",
    "hydrophobicity_ENGD860101.G1","hydrophobicity_ENGD860101.G2","hydrophobicity_ENGD860101.G3",
    "hydrophobicity_FASG890101.G1","hydrophobicity_FASG890101.G2","hydrophobicity_FASG890101.G3",
    "normwaalsvolume.G1","normwaalsvolume.G2","normwaalsvolume.G3",
    "polarity.G1","polarity.G2","polarity.G3",
    "polarizability.G1","polarizability.G2","polarizability.G3",
    "charge.G1","charge.G2","charge.G3",
    "secondarystruct.G1","secondarystruct.G2","secondarystruct.G3",
    "solventaccess.G1","solventaccess.G2","solventaccess.G3"
]



pca_loadings = pca.components_.T  # shape: (n_features, n_components)
pca_loadings_df = pd.DataFrame(
    pca_loadings,
    columns=['PCA1', 'PCA2', 'PCA3'],
    index=feature_names
)

print(pca_loadings_df)


pca_loadings_df.to_csv('pca_loadings_train.csv')


mic_train_df = pd.read_csv('mic_hemo_train.csv')
pca_df = pd.DataFrame(pca_result_train, columns=['PCA1', 'PCA2', 'PCA3'])
mic_hemo_pca_df = pd.concat([mic_train_df, pca_df], axis=1)
mic_hemo_pca_df.to_csv('p_train.csv', index=False)

scaler = StandardScaler()
scaled_data_test = scaler.fit_transform(test_data)


pca = PCA(n_components=3)
pca_result_test = pca.fit_transform(scaled_data_test)
explained_var_test = pca.explained_variance_ratio_
print("Test explained variance ratio (PCA1–3):", explained_var_test)
print("Total explained variance (test):", np.sum(explained_var_test))

mic_test_df = pd.read_csv('mic_hemo_test.csv')
pca_df = pd.DataFrame(pca_result_test, columns=['PCA1', 'PCA2', 'PCA3'])
mic_hemo_pca_df = pd.concat([mic_test_df, pca_df], axis=1)
mic_hemo_pca_df.to_csv('p_test.csv', index=False)

pca_df_train = pd.DataFrame(pca_result_train, columns=['PCA1', 'PCA2', 'PCA3'])
pca_df_test = pd.DataFrame(pca_result_test, columns=['PCA1', 'PCA2', 'PCA3'])

pca_df_train.to_csv('pca_vec_train.csv', index=False)
pca_df_test.to_csv('pca_vec_test.csv', index=False)


label_col = 'mic'
print(mic_train_df.columns)

# def plot_pca_3d(pca_df, labels, title='PCA 3D Projection'):
#     fig = plt.figure(figsize=(8, 6))
#     ax = fig.add_subplot(111, projection='3d')
#

#     scatter = ax.scatter(
#         pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'],
#         c=labels, cmap='coolwarm', edgecolor='k', s=40, alpha=0.8
#     )
#     # scatter = ax.scatter(
#     #     pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'],
#     #     cmap='coolwarm', edgecolor='k', s=40, alpha=0.8
#     # )

#     ax.set_xlabel('PCA1')
#     ax.set_ylabel('PCA2')
#     ax.set_zlabel('PCA3')
#     ax.set_title(title)

#     legend = fig.colorbar(scatter, ax=ax, shrink=0.5)
#     legend.set_label(label_col)
#
#     plt.tight_layout()
#     plt.savefig(title,dpi=300)
#     plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_pca_3d(pca_df, labels, label_col='Label', title='PCA 3D Projection'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        pca_df['PCA1'], pca_df['PCA2'], pca_df['PCA3'],
        c=labels, cmap='coolwarm', edgecolor='k', s=40, alpha=0.8
    )

    ax.set_xlabel('PCA1', fontsize=14, labelpad=10)
    ax.set_ylabel('PCA2', fontsize=14, labelpad=10)
    ax.set_zlabel('PCA3', fontsize=14, labelpad=10)
    ax.set_title(title, fontsize=16, pad=20)

    ax.tick_params(axis='both', labelsize=12)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.1, pad=0.10)
    cbar.set_label(label_col, fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


plot_pca_3d(pca_df_train, mic_train_df[label_col], title='Train PCA Projection1')


plot_pca_3d(pca_df_test, mic_test_df[label_col], title='Test PCA Projection1')
