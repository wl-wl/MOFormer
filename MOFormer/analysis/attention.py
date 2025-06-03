import torch
import torch.nn as nn
import joblib
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from model.Condition import *
from model.Models import get_model
from Embed import Embedder, PositionalEncoder
from matplotlib.gridspec import GridSpec

import pickle
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

def load_pickle_file(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
def plot_attention_heatmap2(attention_weights, x_labels, y_labels):
    attn_weights_np = attention_weights.detach().numpy()
    plt.figure(figsize=(8, 6))


    norm = plt.Normalize(attn_weights_np.min(), attn_weights_np.max())


    for i in range(attn_weights_np.shape[0]):
        for j in range(attn_weights_np.shape[1]):
            size = (attn_weights_np[i, j] * 1000) ** 1.5
            plt.scatter(j, i, s=size, c=[attn_weights_np[i, j]], cmap='Blues', norm=norm, edgecolors='w')

    sm = plt.cm.ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])
    plt.colorbar(sm)


    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, rotation=90)
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels)


    plt.gca().invert_yaxis()
    plt.savefig("8_3.png", dpi=300)
    plt.show()


def plot_attention_heatmap3(attention_weights, x_labels, y_labels):
    attn_weights_np = attention_weights.detach().numpy()
    plt.figure(figsize=(8, 6))

    norm = plt.Normalize(attn_weights_np.min(), attn_weights_np.max())

    colors = [(1, 0, 0), (0, 0, 1)]
    cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors)

    cmap_custom = mcolors.LinearSegmentedColormap.from_list("mycmap", ["#5454EB", "#8F8FFC", "#E8CCE4", "#F4C1CE", "#FB7374", "#F33A3B"])

    # F33A3B"])
    for i in range(attn_weights_np.shape[0]):
        for j in range(attn_weights_np.shape[1]):
            size = (attn_weights_np[i, j] * 1000) ** 1.2
            plt.scatter(j, i, s=size, c=[attn_weights_np[i, j]], cmap=cmap_custom, norm=norm, edgecolors='w')

    sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm)
    sm.set_array([])
    plt.colorbar(sm)

    plt.xticks(ticks=np.arange(len(x_labels)), labels=x_labels, fontsize=14)
    plt.yticks(ticks=np.arange(len(y_labels)), labels=y_labels, fontsize=14)

    plt.gca().invert_yaxis()
    plt.savefig("8_2.png", dpi=300)
    plt.show()


def plot_attention_heatmap(attention_weights, x_labels, y_labels):
    attn_weights_np = attention_weights.detach().numpy()
    plt.figure(figsize=(8, 6))

    norm = plt.Normalize(attn_weights_np.min(), attn_weights_np.max())

    for i in range(attn_weights_np.shape[0]):
        for j in range(attn_weights_np.shape[1]):
            if i <= j:
                # color = 'BuPu' if i == j else 'blue'
                size = (attn_weights_np[i, j] * 1000) ** 1.2
                plt.scatter(j, i, s=size, c=[attn_weights_np[i, j]], cmap='BuPu', norm=norm, edgecolors='w')
            # if i==j:
            #     #4f0e51
            #     size = (np.max(attn_weights_np) * 1000) ** 1.2
            #     plt.scatter(j, i, s=size, c="#4f0e51", cmap='BuPu', norm=norm, edgecolors='w')

    sm = plt.cm.ScalarMappable(cmap='BuPu', norm=norm)
    sm.set_array([])
    plt.colorbar(sm)

    plt.axis('off')

    label_distance = 0.7  # This can be adjusted as needed for better alignment
    for i, label in enumerate(y_labels):
        plt.text(i - label_distance, i + label_distance, label,
                 ha='right', va='bottom', fontsize=11, rotation=0)

    # Adjust top labels to avoid overlapping with the data
    top_label_distance = len(x_labels) * 0.07  # Adjust as needed for label spacing
    for j, label in enumerate(x_labels):
        plt.text(j, -top_label_distance, label,
                 ha='center', va='top', fontsize=11, rotation=0)

    plt.gca().invert_yaxis()
    plt.savefig("1_2.png", dpi=300)


plt.show()


def get_top_n_attention_values(attn_weights, n=4):
    flattened_weights = attn_weights.flatten()
    top_n_indices = np.argsort(flattened_weights)[-n:]
    top_n_values = flattened_weights[top_n_indices]
    return top_n_indices, top_n_values



def print_top_n_attention_values(attn_weights, x_labels, y_labels, n=4):
    top_n_indices, top_n_values = get_top_n_attention_values(attn_weights, n)

    print("Top {} Attention Values:".format(n))
    for idx, value in zip(top_n_indices, top_n_values):
        row_idx, col_idx = np.unravel_index(idx, attn_weights.shape)
        x_value = x_labels[row_idx]
        y_value = y_labels[col_idx]
        print("x: {}, y: {}, value: {}".format(x_value, y_value, value))


def main():
    parser = argparse.ArgumentParser()
    # Data settings
    parser.add_argument('-imp_test', type=bool, default=True)
    parser.add_argument('-src_data', type=str, default='/dataFinal/train.txt')
    parser.add_argument('-src_data_te', type=str, default='/dataFinal/test.txt')
    parser.add_argument('-trg_data', type=str, default='/dataFinal/train.txt')
    parser.add_argument('-trg_data_te', type=str, default='/dataFinal/test.txt')
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-calProp', type=bool, default=False)  # if prop_temp.csv and prop_temp_te.csv exist, set False

    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-no_cuda', type=str, default=False)
    # parser.add_argument('-lr_scheduler', type=str, default="SGDR", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_scheduler', type=str, default="WarmUpDefault", help="WarmUpDefault, SGDR")
    parser.add_argument('-lr_WarmUpSteps', type=int, default=8000, help="only for WarmUpDefault")  # 8000
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-lr_beta1', type=float, default=0.9)
    parser.add_argument('-lr_beta2', type=float, default=0.98)
    parser.add_argument('-lr_eps', type=float, default=1e-9)
    parser.add_argument("--reparam_dropout_rate", type=float, default=0.2, help="dropout rate for reparameterization dropout")

    # KL Annealing
    parser.add_argument('-use_KLA', type=bool, default=True)
    parser.add_argument('-KLA_ini_beta', type=float, default=0.02)
    parser.add_argument('-KLA_inc_beta', type=float, default=0.02)
    parser.add_argument('-KLA_max_beta', type=float, default=1.0)
    parser.add_argument('-KLA_beg_epoch', type=int, default=1)  # KL annealing begin

    # Network sturucture
    parser.add_argument('-use_cond2dec', type=bool, default=True)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument('-cond_dim', type=int, default=5)
    parser.add_argument('-d_model', type=int, default=512)  # 512
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)  # 8
    parser.add_argument('-dropout', type=int, default=0.3)
    parser.add_argument('-batchsize', type=int, default=64)
    # parser.add_argument('-batchsize', type=int, default=1024*8)
    parser.add_argument('-max_strlen', type=int, default=60)  # max 80

    # History
    parser.add_argument('-verbose', type=bool, default=False)
    parser.add_argument('-save_folder_name', type=str, default='saved_model')
    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-printevery', type=int, default=5)
    parser.add_argument('-historyevery', type=int, default=5)  # must be a multiple of printevery
    parser.add_argument('-load_weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=0)

    opt = parser.parse_args()
    opt.device = 2 if opt.no_cuda is False else -1



    # input_data = torch.rand(10, 16)

    # SRC, TRG = create_fields(opt)
    file_path1 = "/save_mic_hemo/SRC.pkl"
    file_path2 = "/save_mic_hemo/TRG.pkl"
    with open(file_path1, 'rb') as file:
        SRC = pickle.load(file)
    with open(file_path2, 'rb') as file:
        TRG = pickle.load(file)
    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    # seq = ['GIHDILKYGKPS']
    # print(len(SRC.vocab),len(TRG.vocab))
    # embed_sentence = Embedder(SRC.vocab, opt.d_model)
    # input_data = embed_sentence(seq)

    model.load_state_dict(torch.load('/save_mic_hemo/model_weights'))
    print(model)

    scaler = joblib.load('/save_mic_hemo/scaler.pkl')

    # scaler = load_pickle_file('/tmp/pycharm_project_PepCPT/save_mic_hemo/scaler.pkl')
    src_tokenizer = load_pickle_file('/save_mic_hemo/SRC.pkl')
    trg_tokenizer = load_pickle_file('/save_mic_hemo/TRG.pkl')

    # seq_data="SIITMTKEAKLPQSWKQIACRLYNTC"
    # cond_data="1.3087247610092163,0.3937290012836456,-0.3321441525720076,-0.8168365444215114,-0.9407572510500508"
    protein_sequence = "RGGRLCYCRRRFCVCT"
    x_labels = list(protein_sequence)
    y_labels = list(protein_sequence)
    input_sequence = [SRC.vocab.stoi[char] for char in protein_sequence]
    input_sequence = torch.LongTensor(input_sequence)
    model.eval()
    cond_input = np.array([[-0.452, 0.283, -2.82312189e+00, -3.64108715e+00, 2.26504774e-01]])
    cond_input = scaler.transform(cond_input)
    cond_input = torch.Tensor(cond_input)
    mask = None


    input_sequence = input_sequence.unsqueeze(0)
    # cond_input = cond_input.unsqueeze(0)
    print(input_sequence.shape, cond_input.shape)
    # 调用编码器
    with torch.no_grad():
        # encoder_output, mu, log_var = model.encoder(input_sequence, cond_input, mask)
        cond2enc = model.encoder.embed_cond2enc(cond_input).view(cond_input.size(0), cond_input.size(1), -1)
        x = model.encoder.embed_sentence.embed(input_sequence)
        # cond2enc = torch.unsqueeze(cond2enc, 0)

        # print(cond2enc.shape, x.shape)
        x = torch.cat([cond2enc, x], dim=1)

        for layer in model.encoder.layers:

            multihead_attn = layer.attn
            Q = layer.attn.q_linear(x)
            K = layer.attn.k_linear(x)


            attn_weights = torch.matmul(Q, K.transpose(-2, -1))  # [batch_size, n_heads, seq_length, seq_length]
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            N = len(x_labels)
            print_top_n_attention_values(attn_weights[0][:N, :N], x_labels, y_labels)

            plot_attention_heatmap3(attn_weights[0][:N, :N], x_labels=x_labels, y_labels=y_labels)
            plot_attention_heatmap2(attn_weights[0][N:, N:], x_labels=["MIC", "HEMO", "PCA1", "PCA2", "PCA3"], y_labels=["MIC", "HEMO", "PCA1", "PCA2", "PCA3"])

    print(attn_weights[0].shape)



if __name__ == "__main__":
    main()
