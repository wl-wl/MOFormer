import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns
import pandas as pd


def checkdata(fpath):
    results = pd.read_csv(fpath)

    # MIC, HEMO, PCA1 = results.iloc[:, 0], results.iloc[:, 1], results.iloc[:, 2]
    MIC, HEMO,PCA1,PCA2,PCA3 = results.iloc[:, 0], results.iloc[:, 1],results.iloc[:, 2], results.iloc[:, 3], results.iloc[:, 4]
    # figure, ((ax1,ax2,ax3)) = plt.subplots(nrows=1, ncols=3)
    figure, ((ax1, ax2,ax3,ax4,ax5)) = plt.subplots(nrows=1, ncols=5)

    sns.violinplot(y = "MIC", data =results, ax=ax1, color=sns.color_palette()[0])
    sns.violinplot(y = "HEMO", data =results, ax=ax2, color=sns.color_palette()[1])
    sns.violinplot(y = "PCA1", data =results, ax=ax3, color=sns.color_palette()[2])
    sns.violinplot(y="PCA2", data=results, ax=ax4, color=sns.color_palette()[3])
    sns.violinplot(y="PCA3", data=results, ax=ax5, color=sns.color_palette()[4])

    ax1.set(xlabel='MIC', ylabel='')
    ax2.set(xlabel='HEMO', ylabel='')
    ax3.set(xlabel='PCA1', ylabel='')
    ax4.set(xlabel='PCA2', ylabel='')
    ax5.set(xlabel='PCA3', ylabel='')

    bound_MIC = get_quatiles(MIC)
    for i in range(4):
        text = ax1.text(0, bound_MIC[i], f'{bound_MIC[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_HEMO = get_quatiles(HEMO)
    for i in range(4):
        text = ax2.text(0, bound_HEMO[i], f'{bound_HEMO[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_PCA1 = get_quatiles(PCA1)
    for i in range(4):
        text = ax3.text(0, bound_PCA1[i], f'{bound_PCA1[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_PCA2 = get_quatiles(PCA2)
    for i in range(4):
        text = ax4.text(0, bound_PCA2[i], f'{bound_PCA2[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    bound_PCA3 = get_quatiles(PCA3)
    for i in range(4):
        text = ax5.text(0, bound_PCA3[i], f'{bound_PCA3[i]:.2f}', ha='right', va='center', fontweight='bold', size=10, color='white')
        text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal(), ])

    # plt.show()

    MIC_max, MIC_min = min(bound_MIC[0], MIC.max()), max(bound_MIC[-1], MIC.min())
    HEMO_max, HEMO_min = min(bound_HEMO[0], HEMO.max()), max(bound_HEMO[-1], HEMO.min())
    PCA1_max, PCA1_min = min(bound_PCA1[0], PCA1.max()), max(bound_PCA1[-1], PCA1.min())
    PCA2_max, PCA2_min = min(bound_PCA2[0], PCA2.max()), max(bound_PCA2[-1], PCA2.min())
    PCA3_max, PCA3_min = min(bound_PCA3[0], PCA3.max()), max(bound_PCA3[-1], PCA3.min())


    return MIC_max, MIC_min, HEMO_max, HEMO_min, PCA1_max, PCA1_min,PCA2_max, PCA2_min,PCA3_max, PCA3_min
    # return MIC_max, MIC_min, HEMO_max, HEMO_min

def get_quatiles(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    UAV = Q3 + 1.5 * IQR
    LAV = Q1 - 1.5 * IQR
    return [UAV, Q3, Q1, LAV]