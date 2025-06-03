import argparse
import torch.nn as nn
import time
import torch
import numpy as np
from model.Models import get_model
from model.Condition import *
import torch.nn.functional as F
from model.Optimizer import CosineWithRestarts
from model.Batch import create_masks
from sklearn.preprocessing import RobustScaler, StandardScaler
import joblib
import dill as pickle
import pandas as pd
import csv
import timeit


def KLAnnealer(opt, epoch):
    beta = opt.KLA_ini_beta + opt.KLA_inc_beta * ((epoch + 1) - opt.KLA_beg_epoch)
    return beta


def loss_function_constrative(opt, z, z1):
    z = torch.sum(z, dim=1)
    z1 = torch.sum(z1, dim=1)

    cossim = nn.CosineSimilarity(dim=-1).to(opt.device)
    sim11 = cossim(z.unsqueeze(-2), z.unsqueeze(-3)) / 0.5
    sim22 = cossim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / 0.5
    sim12 = cossim(z.unsqueeze(-2), z1.unsqueeze(-3)) / 0.5

    d = sim12.shape[-1]
    sim11[..., range(d), range(d)] = float('-inf')
    sim22[..., range(d), range(d)] = float('-inf')

    raw_scores1 = torch.cat([sim12, sim11], dim=-1)
    raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)

    logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
    labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)

    criterion = nn.CrossEntropyLoss().to(opt.device)
    nce_loss = criterion(logits, labels)
    # nce_loss=0
    return nce_loss


def loss_function(opt, beta, preds_prop, preds_pep, ys_cond, ys_pep, mu, log_var):
    RCE_pep = F.cross_entropy(preds_pep.contiguous().view(-1, preds_pep.size(-1)), ys_pep, ignore_index=opt.trg_pad, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    if opt.use_cond2dec == True:
        RCE_prop = F.mse_loss(preds_prop, ys_cond, reduction='sum')
        loss = RCE_pep + RCE_prop + beta * KLD

        preds_prop_temp = preds_prop.squeeze(-1)
        preds_prop_temp = preds_prop_temp.to(opt.device)
        loss_part_1 = F.mse_loss(preds_prop_temp[:, 0].mean(), torch.tensor(-1.0).to(opt.device))
        mask = preds_prop_temp[:, 1] > 0.5
        # loss_part_2 = mask.float() * (preds_prop_temp[:, 1] - 0.5) ** 2

        loss_part_2 = F.relu(preds_prop_temp[:, 1] - 0.5)
        loss_part_2 = loss_part_2.sum()
        loss = RCE_pep + RCE_prop + beta * KLD + loss_part_1 + loss_part_2
    else:
        RCE_prop = torch.zeros(1)
        loss = RCE_pep + beta * KLD
    return loss, RCE_pep, RCE_prop, KLD


def train_model(model, opt):
    print("training model...")
    global robustScaler
    model.train()

    start = time.time()
    if opt.checkpoint > 0:
        cptime = time.time()

    history_epoch, history_beta, history_lr = [], [], []
    history_total_loss, history_RCE_pep_loss, history_RCE_prop_loss, history_KLD_loss = [], [], [], []
    history_total_loss_te, history_RCE_pep_loss_te, history_RCE_prop_loss_te, history_KLD_loss_te = [], [], [], []

    beta = 0
    current_step = 0
    for epoch in range(opt.epochs):
        total_loss, RCE_pep_loss, RCE_prop_loss, KLD_loss = 0, 0, 0, 0
        total_loss_te, RCE_pep_loss_te, RCE_prop_loss_te, KLD_loss_te = 0, 0, 0, 0
        total_loss_accum_te, RCE_pep_loss_accum_te, RCE_prop_loss_accum_te, KLD_loss_accum_te = 0, 0, 0, 0
        accum_train_printevery_n, accum_test_n, accum_test_printevery_n = 0, 0, 0

        if opt.floyd is False:
            print("     {TR}   %dm: epoch %d [%s]  %d%%  loss = %s" % \
                  ((time.time() - start) // 60, epoch + 1, "".join(' ' * 20), 0, '...'), end='\r')

        if opt.checkpoint > 0:
            torch.save(model.state_dict(), 'weights/model_weights')

        # KL annealing
        if opt.use_KLA == True:
            if epoch + 1 >= opt.KLA_beg_epoch and beta < opt.KLA_max_beta:
                beta = KLAnnealer(opt, epoch)
        else:
            beta = 1

        for i, batch in enumerate(opt.train):
            current_step += 1
            src = batch.src.transpose(0, 1).to('cuda')
            trg = batch.trg.transpose(0, 1).to('cuda')
            trg_input = trg[:, :-1]

            cond = torch.stack([batch.MIC, batch.HEMO, batch.PCA1, batch.PCA2, batch.PCA3]).transpose(0, 1).to('cuda')

            src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
            preds_prop, preds_pep, mu, log_var, z, preds_prop1, preds_pep1, mu1, log_var1, z1 = model(src, trg_input, cond, src_mask, trg_mask)
            ys_pep = trg[:, 1:].contiguous().view(-1)
            ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

            opt.optimizer.zero_grad()

            loss, RCE_pep, RCE_prop, KLD = loss_function(opt, beta, preds_prop, preds_pep, ys_cond, ys_pep, mu, log_var)
            con_loss = loss_function_constrative(opt, z, z1)

            loss += con_loss
            loss.backward()
            opt.optimizer.step()
            if opt.lr_scheduler == "SGDR":
                opt.sched.step()

            if opt.lr_scheduler == "WarmUpDefault":
                head = np.float(np.power(np.float(current_step), -0.5))
                tail = np.float(current_step) * np.power(np.float(opt.lr_WarmUpSteps), -1.5)
                lr = np.float(np.power(np.float(opt.d_model), -0.5)) * min(head, tail)
                for param_group in opt.optimizer.param_groups:
                    param_group['lr'] = lr

            for param_group in opt.optimizer.param_groups:
                current_lr = param_group['lr']

            total_loss += loss.item()
            RCE_pep_loss += RCE_pep.item()
            RCE_prop_loss += RCE_prop.item()
            KLD_loss += KLD.item()

            accum_train_printevery_n += len(batch)
            if (i + 1) % opt.printevery == 0:
                p = int(100 * (i + 1) / opt.train_len)
                avg_loss = total_loss / accum_train_printevery_n
                avg_RCE_pep_loss = RCE_pep_loss / accum_train_printevery_n
                avg_RCE_prop_loss = RCE_prop_loss / accum_train_printevery_n
                avg_KLD_loss = KLD_loss / accum_train_printevery_n
                if (i + 1) % (opt.historyevery) == 0:
                    history_epoch.append(epoch + 1)
                    history_beta.append(beta)
                    for param_group in opt.optimizer.param_groups:
                        history_lr.append(param_group['lr'])
                    history_total_loss.append(avg_loss)
                    history_RCE_pep_loss.append(avg_RCE_pep_loss)
                    history_RCE_prop_loss.append(avg_RCE_prop_loss)
                    history_KLD_loss.append(avg_KLD_loss)

                if opt.floyd is False:
                    print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_pep = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" % (
                    (time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))), p, avg_loss, avg_RCE_pep_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr), end='\r')
                else:
                    print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_pep = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f, lr = %.6f" % \
                          ((time.time() - start) // 60, epoch + 1, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))), p, avg_loss, avg_RCE_pep_loss, avg_RCE_prop_loss, avg_KLD_loss, beta, current_lr))
                accum_train_printevery_n, total_loss, RCE_pep_loss, RCE_prop_loss, KLD_loss = 0, 0, 0, 0, 0

            if opt.checkpoint > 0 and ((time.time() - cptime) // 60) // opt.checkpoint >= 1:
                torch.save(model.state_dict(), 'weights/model_weights')
                cptime = time.time()

        print("     {TR}   %dm: epoch %d [%s%s]  %d%%  loss = %.3f, RCE_pep = %.3f, RCE_prop = %.5f, KLD = %.5f, con_loss = %.5f, beta = %.4f, lr = %.6f" % \
              ((time.time() - start) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss, avg_RCE_pep_loss, avg_RCE_prop_loss, avg_KLD_loss, con_loss, beta, current_lr))

        # Test
        if opt.imp_test == True:
            model.eval()

            if opt.floyd is False:
                print("     {TE}   %dm:         [%s]  %d%%  loss = %s" % \
                      ((time.time() - start) // 60, "".join(' ' * 20), 0, '...'), end='\r')

            with torch.no_grad():
                for i, batch in enumerate(opt.test):
                    src = batch.src.transpose(0, 1).to('cuda')
                    trg = batch.trg.transpose(0, 1).to('cuda')
                    trg_input = trg[:, :-1]
                    cond = torch.stack([batch.MIC, batch.HEMO, batch.PCA1, batch.PCA2, batch.PCA3]).transpose(0, 1).to('cuda')
                    # cond = torch.stack([batch.MIC, batch.HEMO]).transpose(0, 1).to('cuda')
                    src_mask, trg_mask = create_masks(src, trg_input, cond, opt)
                    preds_prop, preds_pep, mu, log_var, z, preds_prop1, preds_pep1, mu1, log_var1, z1 = model(src, trg_input, cond, src_mask, trg_mask)
                    ys_pep = trg[:, 1:].contiguous().view(-1)
                    ys_cond = torch.unsqueeze(cond, 2).contiguous().view(-1, opt.cond_dim, 1)

                    loss_te, RCE_pep_te, RCE_prop_te, KLD_te = loss_function(opt, beta, preds_prop, preds_pep, ys_cond, ys_pep, mu, log_var)
                    con_loss = loss_function_constrative(opt, z, z1)

                    total_loss_te += loss_te.item()
                    RCE_pep_loss_te += RCE_pep_te.item()
                    RCE_prop_loss_te += RCE_prop_te.item()
                    KLD_loss_te += KLD_te.item()
                    total_loss_accum_te += loss_te.item()
                    RCE_pep_loss_accum_te += RCE_pep_te.item()
                    RCE_prop_loss_accum_te += RCE_prop_te.item()
                    KLD_loss_accum_te += KLD_te.item()

                    accum_test_n += len(batch)
                    accum_test_printevery_n += len(batch)
                    if (i + 1) % opt.printevery == 0:
                        p = int(100 * (i + 1) / opt.test_len)
                        avg_loss_te = total_loss_te / accum_test_printevery_n
                        avg_RCE_pep_loss_te = RCE_pep_loss_te / accum_test_printevery_n
                        avg_RCE_prop_loss_te = RCE_prop_loss_te / accum_test_printevery_n
                        avg_KLD_loss_te = KLD_loss_te / accum_test_printevery_n

                        if opt.floyd is False:
                            print("     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_pep = %.3f, RCE_prop = %.5f, KLD = %.5f, beta = %.4f" % \
                                  ((time.time() - start) // 60, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))), p, avg_loss_te, avg_RCE_pep_loss_te, avg_RCE_prop_loss_te, avg_KLD_loss_te, beta), end='\r')
                        else:
                            print("     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_pep = %.3f, RCE_prop = %.3f, KLD = %.5f, beta = %.4f" % \
                                  ((time.time() - start) // 60, "".join('#' * (p // 5)), "".join(' ' * (20 - (p // 5))), p, avg_loss_te, avg_RCE_pep_loss_te, avg_RCE_prop_loss_te, avg_KLD_loss_te, beta))
                        accum_test_printevery_n, total_loss_te, RCE_pep_loss_te, RCE_prop_loss_te, KLD_loss_te = 0, 0, 0, 0, 0

                print("     {TE}   %dm:         [%s%s]  %d%%  loss = %.3f, RCE_pep = %.3f, RCE_prop = %.5f, KLD = %.5f, con_loss = %.5f, beta = %.4f\n" % \
                      ((time.time() - start) // 60, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss_te, avg_RCE_pep_loss_te, avg_RCE_prop_loss_te, avg_KLD_loss_te, con_loss, beta))

            if epoch == 0:
                opt.margin = len(history_epoch)

            for j in range(opt.margin):
                history_total_loss_te.append("")
                history_RCE_pep_loss_te.append("")
                history_RCE_prop_loss_te.append("")
                history_KLD_loss_te.append("")
            history_epoch.append(epoch + 1)
            history_lr.append(current_lr)
            history_beta.append(beta)
            history_total_loss_te.append(total_loss_accum_te / len(opt.test.dataset))
            history_RCE_pep_loss_te.append(RCE_pep_loss_accum_te / len(opt.test.dataset))
            history_RCE_prop_loss_te.append(RCE_prop_loss_accum_te / len(opt.test.dataset))
            history_KLD_loss_te.append(KLD_loss_accum_te / len(opt.test.dataset))
        history_total_loss.append(avg_loss)
        history_RCE_pep_loss.append(avg_RCE_pep_loss)
        history_RCE_prop_loss.append(avg_RCE_prop_loss)
        history_KLD_loss.append(avg_KLD_loss)

        # Export train/test history
        if opt.imp_test == True:
            history = pd.DataFrame(
                {"epochs": history_epoch, "beta": history_beta, "lr": history_lr, "total_loss": history_total_loss, "total_loss_te": history_total_loss_te,
                 "RCE_pep_loss": history_RCE_pep_loss, "RCE_pep_loss_te": history_RCE_pep_loss_te,
                 "RCE_prop_loss": history_RCE_prop_loss, "RCE_prop_loss_te": history_RCE_prop_loss_te,
                 "KLD_loss": history_KLD_loss, "KLD_loss_te": history_KLD_loss_te})
            history.to_csv('trHist_lat={}_epo={}_{}.csv'.format(opt.latent_dim, opt.epochs, time.strftime("%Y%m%d")), index=True)
        else:
            history = pd.DataFrame(
                {"epochs": history_epoch, "beta": history_beta, "lr": history_lr, "total_loss": history_total_loss, "RCE_pep_loss": history_RCE_pep_loss,
                 "RCE_prop_loss": history_RCE_prop_loss, "KLD_loss": history_KLD_loss})
            history.to_csv('trHist_lat={}_epo={}_{}.csv'.format(opt.latent_dim, opt.epochs, time.strftime("%Y%m%d")), index=True)

    torch.save(model.state_dict(), 'transformer_pep_model.pth')
    # Export weights every epoch
    # if not os.path.isdir('{}'.format(opt.save_folder_name)):
    #     os.mkdir('{}'.format(opt.save_folder_name))
    # if not os.path.isdir('{}/epo{}'.format(opt.save_folder_name, epoch + 1)):
    #     os.mkdir('{}/epo{}'.format(opt.save_folder_name, epoch + 1))
    # torch.save_mic_hemo(model.state_dict(), f'{opt.save_folder_name}/epo{epoch+1}/model_weights')
    # joblib.dump(robustScaler, f'{opt.save_folder_name}/epo{epoch+1}/scaler.pkl')


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

    # Learning hyperparameters
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
    opt.device = 0 if opt.no_cuda is False else -1

    if opt.historyevery % opt.printevery != 0:
        raise ValueError("historyevery must be a multiple of printevery: {} % {} != 0".format(opt.historyevery, opt.printevery))

    if opt.device == 0:
        assert torch.cuda.is_available()
    print(torch.cuda.is_available(), opt.device)
    read_data(opt)



    PROP, PROP_te = pd.read_csv("/dataFinal/p_train.csv"), pd.read_csv("/dataFinal/p_test.csv")

    SRC, TRG = create_fields(opt)
    opt.max_MIC, opt.min_MIC, opt.max_HEMO, opt.min_HEMO, opt.max_PCA1, opt.min_PCA1, opt.max_PCA2, opt.min_PCA2, opt.max_PCA3, opt.min_PCA3 \
        = PROP["MIC"].max(), PROP["MIC"].min(), PROP["HEMO"].max(), PROP["HEMO"].min(), PROP_te["PCA1"].max(), PROP_te["PCA1"].min(), PROP_te["PCA2"].max(), PROP_te["PCA2"].min(), PROP_te["PCA3"].max(), PROP_te["PCA3"].min()
    # opt.max_MIC, opt.min_MIC, opt.max_HEMO, opt.min_HEMO \
    #     = PROP["MIC"].max(), PROP["MIC"].min(), PROP["HEMO"].max(), PROP["HEMO"].min()

    robustScaler = RobustScaler()
    robustScaler.fit(PROP)
    # if not os.path.isdir('{}'.format(opt.save_folder_name)):
    #     os.mkdir('{}'.format(opt.save_folder_name))
    # joblib.dump(robustScaler, 'scaler.pkl')
    # robustScaler = joblib.load('scaler.pkl')

    PROP, PROP_te = pd.DataFrame(robustScaler.transform(PROP)), pd.DataFrame(robustScaler.transform(PROP_te))

    opt.train = create_dataset(opt, SRC, TRG, PROP, tr_te='tr')
    opt.test = create_dataset(opt, SRC, TRG, PROP_te, tr_te='te')

    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of trainable parameters: {}".format(total_trainable_params))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.lr_beta1, opt.lr_beta2), eps=opt.lr_eps)
    if opt.lr_scheduler == "SGDR":
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/" % (opt.checkpoint))

    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))

    train_model(model, opt)

    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG, robustScaler)


def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response


def promptNextAction(model, opt, SRC, TRG, robustScaler):
    dst = 'save_mic_hemo'
    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    print("saving weights to " + dst + "/...")
    torch.save(model.state_dict(), f'{dst}/model_weights')
    if saved_once == 0:
        pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
        pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
        joblib.dump(robustScaler, open(f'{dst}/scaler.pkl', 'wb'))

    # saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    #
    # if opt.load_weights is not None:
    #     dst = opt.load_weights
    # if opt.checkpoint > 0:
    #     dst = 'weights'
    #
    # while True:
    #     save_mic_hemo = yesno(input('training complete, save_mic_hemo results? [y/n] : '))
    #     if save_mic_hemo == 'y':
    #         while True:
    #             if saved_once != 0:
    #                 res = yesno("save_mic_hemo to same folder? [y/n] : ")
    #                 if res == 'y':
    #                     break
    #             dst = input('enter folder name to create for weights (no spaces) : ')
    #             if ' ' in dst or len(dst) < 1 or len(dst) > 30:
    #                 dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
    #             else:
    #                 try:
    #                     os.mkdir(dst)
    #                 except:
    #                     res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
    #                     if res == 'n':
    #                         continue
    #                 break
    #
    #         print("saving weights to " + dst + "/...")
    #         torch.save_mic_hemo(model.state_dict(), f'{dst}/model_weights')
    #         if saved_once == 0:
    #             pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
    #             pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
    #             joblib.dump(robustScaler, open(f'{dst}/scaler.pkl', 'wb'))
    #             saved_once = 1
    #
    #         print("weights and field pickles saved to " + dst)
    #
    #     res = yesno(input("train for more epochs? [y/n] : "))
    #     if res == 'y':
    #         while True:
    #             epochs = input("type number of epochs to train for : ")
    #             try:
    #                 epochs = int(epochs)
    #             except:
    #                 print("input not a number")
    #                 continue
    #             if epochs < 1:
    #                 print("epochs must be at least 1")
    #                 continue
    #             else:
    #                 break
    #         opt.epochs = epochs
    #         train_model(model, opt)
    #     else:
    #         print("exiting program...")
    #         break


if __name__ == "__main__":
    main()

