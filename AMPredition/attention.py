import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from Process import *
from Models import get_model
from Embed import Embedder, PositionalEncoder

def plot_attention_heatmap(attention_weights, x_labels, y_labels):
    # 将 PyTorch 张量转换为 NumPy 数组
    attention_weights_np = attention_weights.detach().numpy()

    # 使用 seaborn 绘制热力图
    sns.set()
    ax = sns.heatmap(attention_weights_np, cmap="YlGnBu", xticklabels=x_labels, yticklabels=y_labels)

    # 显示图形
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('-imp_test', type=bool, default=True)
    parser.add_argument('-src_data', type=str, default='data/moses/elite_train.txt')
    parser.add_argument('-src_data_te', type=str, default='data/moses/elite_test.txt')
    parser.add_argument('-trg_data', type=str, default='data/moses/elite_train.txt')
    parser.add_argument('-trg_data_te', type=str, default='data/moses/elite_test.txt')
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-calProp', type=bool, default=False) #if prop_temp.csv and prop_temp_te.csv exist, set False

    # Learning hyperparameters
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-no_cuda', type=str, default=False)
    # parser.add_argument('-lr_scheduler', type=str, default="SGDR", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_WarmUpSteps', type=int, default=8000, help="only for WarmUpDefault")
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-lr_beta1', type=float, default=0.9)
    parser.add_argument('-lr_beta2', type=float, default=0.98)
    parser.add_argument('-lr_eps', type=float, default=1e-9)

    # KL Annealing
    parser.add_argument('-use_KLA', type=bool, default=True)
    parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
    parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
    parser.add_argument('-KLA_max_beta', type=float, default=1.0)
    parser.add_argument('-KLA_beg_epoch', type=int, default=1) #KL annealing begin

    # Network sturucture
    parser.add_argument('-use_cond2dec', type=bool, default=True)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-cond_dim', type=int, default=3)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.3)
    parser.add_argument('-batchsize', type=int, default=256)
    # parser.add_argument('-batchsize', type=int, default=1024*8)
    parser.add_argument('-max_strlen', type=int, default=60)  # max 80

    # History
    parser.add_argument('-verbose', type=bool, default=False)
    parser.add_argument('-save_folder_name', type=str, default='saved_model')
    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-printevery', type=int, default=5)
    parser.add_argument('-historyevery', type=int, default=5) # must be a multiple of printevery
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    opt = parser.parse_args()
    opt.device = 0 if opt.no_cuda is False else -1

    # 序列长度以及每一个字符的编码
    seq=''
    embed_sentence = Embedder(opt.vocab_size, opt.d_model)
    input_data = embed_sentence(seq)
    # input_data = torch.rand(10, 16)

    SRC, TRG = create_fields(opt)
    # 创建 Transformer 模型实例
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    # 加载模型参数
    model.load_state_dict(torch.load('transformer_pep_model.pth'))
    model.eval()

    # 前向传播获取注意力得分
    output = model(input_data.unsqueeze(0))  # 添加 batch 维度
    attention_weights = model.transformer_layer.layers[0].attn.output_weights

    # 绘制热力图
    plot_attention_heatmap(attention_weights[0], x_labels=list(range(10)), y_labels=list(range(10)))
