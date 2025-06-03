import sys
from io import StringIO
import argparse
import time
import torch
from Models import get_model
from model.Condition import *
import torch.nn.functional as F
from model.Optimizer import CosineWithRestarts
from model.Batch import create_masks
import pdb
import dill as pickle
import argparse
from rdkit import Chem
from model.Models import get_model
from model.Sample import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import re
import numpy as np
import math
import moses
from model.Rand_gen import rand_gen_from_data_distribution, tokenlen_gen_from_data_distribution
from model.Checkdata import checkdata

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]

    return 0

def gen_pep(cond, model, opt, SRC, TRG, toklen, z):
    model.eval()

    robustScaler = joblib.load(opt.load_weights + '/scaler.pkl')

    if opt.conds == 'm':
        cond = cond.reshape(1, -1)
    elif opt.conds == 's':
        cond = cond.reshape(1, -1)
    elif opt.conds == 'l':
        cond = cond.reshape(1, -1)
    else:
        cond = np.array(cond.split(',')[:-1]).reshape(1, -1)

    cond = robustScaler.transform(cond)
    cond = Variable(torch.Tensor(cond))

    sentence = beam_search(cond, model, SRC, TRG, toklen, opt, z)
    return sentence

def inference(opt, model, SRC, TRG):
    peptides, val_check, conds_trg, conds_rdkit, toklen_check, toklen_gen = [], [], [], [], [], []
    if opt.conds == 'm':
        n_samples = 1000
        nBins = [1000, 50, 50, 50, 50]
        nBins = [100, 100, 100, 100, 100]

        data = pd.read_csv(opt.load_traindata)
        toklen_data = pd.read_csv(opt.load_toklendata)

        conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
        toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=n_samples)

        start = time.time()
        for idx in range(n_samples):
            toklen = int(toklen_data[idx]) + 5  #

            z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
            peptide_tmp = gen_pep(conds[idx], model, opt, SRC, TRG, toklen, z)
            toklen_gen.append(peptide_tmp.count(' ') + 1)
            peptide_tmp = ''.join(peptide_tmp).replace(" ", "")
            print(f"Task {idx + 1}/{n_samples} completed")
            with open('0318_toxi_f.txt', 'a') as f:
                f.write(peptide_tmp + '\n')
            peptides.append(peptide_tmp)
    elif opt.conds == 's':
        print("\nGenerating peptides for 10 condition sets...")
        n_samples = 10
        n_per_samples = 200
        nBins = [1000, 1000, 1000]
        data = pd.read_csv(opt.load_traindata)
        toklen_data = pd.read_csv(opt.load_toklendata)

        conds = rand_gen_from_data_distribution(data, size=n_samples, nBins=nBins)
        toklen_data = tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=n_samples * n_per_samples)

        print("conds:\n", conds)
        start = time.time()
        for idx in range(n_samples):
            for i in range(n_per_samples):
                toklen = int(toklen_data[idx * n_per_samples + i]) + 3  # +3 due to cond2enc
                z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))
                peptide_tmp = gen_pep(conds[idx], model, opt, SRC, TRG, toklen, z)
                toklen_gen.append(peptide_tmp.count(" ") + 1)
                peptide_tmp = ''.join(peptide_tmp).replace(" ", "")

                peptides.append(peptide_tmp)
                conds_trg.append(conds[idx])

                toklen_check.append(toklen - 5)
    else:
        conds = opt.conds.split(';')
        toklen_data = pd.read_csv(opt.load_toklendata)
        toklen = int(tokenlen_gen_from_data_distribution(data=toklen_data, nBins=int(toklen_data.max() - toklen_data.min()), size=1)) + 5  # +3 due to cond2enc

        z = torch.Tensor(np.random.normal(size=(1, toklen, opt.latent_dim)))

        for cond in conds:
            peptides.append(gen_pep(cond + ',', model, opt, SRC, TRG, toklen, z))
        toklen_gen = peptides[0].count(" ") + 1
        peptides = ''.join(peptides).replace(" ", "")
    return peptides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=str, default="save_mic_toxi2")
    parser.add_argument('-load_traindata', type=str, default="/dataFinal_toxi/p_train.csv")
    parser.add_argument('-load_toklendata', type=str, default='toklen_list.csv')
    parser.add_argument('-k', type=int, default=4)
    parser.add_argument('-lang_format', type=str, default='SMILES')
    parser.add_argument('-max_strlen', type=int, default=60)  # max 80
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)

    parser.add_argument('-use_cond2dec', type=bool, default=True)
    parser.add_argument('-use_cond2lat', type=bool, default=True)
    parser.add_argument('-cond_dim', type=int, default=5)
    parser.add_argument('-latent_dim', type=int, default=128)
    parser.add_argument("--reparam_dropout_rate", type=float, default=0.2, help="dropout rate for reparameterization dropout")

    # parser.add_argument('-epochs', type=int, default=1111111111111)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-lr_beta1', type=int, default=0.9)
    parser.add_argument('-lr_beta2', type=int, default=0.98)
    parser.add_argument('-previous_beam', default=None)

    parser.add_argument('-print_model', type=bool, default=False)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')

    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1

    assert opt.k > 0
    assert opt.max_strlen > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.max_MIC, opt.min_MIC, opt.max_HEMO, opt.min_HEMO, opt.max_PCA1, opt.min_PCA1, opt.max_PCA2, opt.min_PCA2, opt.max_PCA3, opt.min_PCA3 = checkdata(opt.load_traindata)

    while True:
        opt.conds = "m"
        if opt.conds == "q":
            break
        if opt.conds == "m":
            peptide = inference(opt, model, SRC, TRG)
            break
        if opt.conds == "s":
            peptide = inference(opt, model, SRC, TRG)
            break
        else:
            peptide = inference(opt, model, SRC, TRG)


if __name__ == '__main__':
    main()
