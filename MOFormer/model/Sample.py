import torch
from Batch import nopeak_mask
import torch.nn.functional as F
import math
import torch.nn as nn
import numpy as np


class SequenceEmbedder(nn.Module):
    def __init__(self, opt, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SequenceEmbedder, self).__init__()
        self.device = opt.device
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(opt.device)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True).to(opt.device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(opt.device)

    def forward(self, x):
        # x: [batch_size, seq_len]
        x = x.to(self.device)
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        # encoder_embedding_weights = model.encoder.embed_sentence.embed.weight.data
        output, hidden = self.rnn(embedded)  # output: [batch_size, seq_len, hidden_dim]
        # output = self.fc(output)  # [batch_size, seq_len, output_dim]
        return output


def compute_gamma(feature_extractor, gamma, beam1):
    features1 = feature_extractor(beam1)
    n_features = features1.size(2)
    flattened_features = features1.flatten()
    variance = flattened_features.var()
    gamma = 1 / (n_features * variance.item())
    return gamma


def euclidean_distance(seq1, seq2):
    seq1, seq2 = torch.tensor(seq1).float(), torch.tensor(seq2).float()

    length = min(len(seq1), len(seq2))
    seq1, seq2 = seq1[:length], seq2[:length]
    return torch.norm(seq1 - seq2)


def smith_waterman(seq1, seq2, match_score=3, mismatch_penalty=-3, gap_penalty=-2):

    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1))
    max_score = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                diag_score = score_matrix[i - 1, j - 1] + match_score
            else:
                diag_score = score_matrix[i - 1, j - 1] + mismatch_penalty

            up_score = score_matrix[i - 1, j] + gap_penalty
            left_score = score_matrix[i, j - 1] + gap_penalty
            score = max(0, diag_score, up_score, left_score)
            score_matrix[i, j] = score

            if score > max_score:
                max_score = score

    return max_score


def gaussian_rbf(x, y, gamma):
    x = torch.tensor(x).float()
    y = torch.tensor(y).float()
    diff = x - y
    diff = diff.view(-1)
    sq_diff = torch.dot(diff, diff)
    return torch.exp(-gamma * sq_diff)


def hausdorff_similarity(feature_extractor, beam1, beam2, gamma):


    features1 = feature_extractor(beam1)
    features2 = feature_extractor(beam2)


    min_sim = float('inf')

    for i in range(features1.size(0)):
        for j in range(features2.size(0)):
            sim = gaussian_rbf(features1[i], features2[j], gamma)
            min_sim = min(min_sim, sim.item())
    # if min_sim !=0.0:
    #     print('min_sim:',min_sim)
    return min_sim


def threshold_hausdorff_similarity(feature_extractor, beam, gamma, j_nearest):
    # beam:new_beam
    features = feature_extractor(beam)
    # features = beam
    similarities = torch.zeros((features.size(0), features.size(0)))

    for i in range(features.size(0)):
        for j in range(features.size(0)):
            if i != j:
                similarities[i, j] = gaussian_rbf(features[i], features[j], gamma)


    densities = torch.zeros(features.size(0))
    for i in range(features.size(0)):
        distances, indices = torch.topk(similarities[i], j_nearest, largest=False)
        dr_index = indices[-1]
        densities[i] = similarities[i, dr_index]


    threshold = 1 / (gamma * densities.min())

    # if threshold!=float('inf'):
    #     print('threshold',threshold)

    return threshold


def init_vars(cond, model, SRC, TRG, toklen, opt, z):
    init_tok = TRG.vocab.stoi['<sos>']

    src_mask = (torch.ones(1, 1, toklen) != 0)
    trg_mask = nopeak_mask(1, opt)

    trg_in = torch.LongTensor([[init_tok]])

    if opt.device == 0:
        trg_in, z, src_mask, trg_mask = trg_in.cuda(), z.cuda(), src_mask.cuda(), trg_mask.cuda()

    if opt.use_cond2dec == True:
        output_pep = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask))[:, 5:, :]
    else:
        output_pep = model.out(model.decoder(trg_in, z, cond, src_mask, trg_mask))
    out_pep = F.softmax(output_pep, dim=-1)

    probs, ix = out_pep[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)

    outputs = torch.zeros(opt.k, opt.max_strlen).long()
    if opt.device == 0:
        outputs = outputs.cuda()
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]

    e_outputs = torch.zeros(opt.k, z.size(-2), z.size(-1))
    if opt.device == 0:
        e_outputs = e_outputs.cuda()
    e_outputs[:, :] = z[0]

    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0, 1)
    k_probs, k_ix = log_probs.view(-1).topk(k)

    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def beam_search(cond, model, SRC, TRG, toklen, opt, z):
    if opt.device == 0:
        cond = cond.cuda()
    cond = cond.view(1, -1)

    outputs, e_outputs, log_scores = init_vars(cond, model, SRC, TRG, toklen, opt, z)
    cond = cond.repeat(opt.k, 1)
    src_mask = (torch.ones(1, 1, toklen) != 0)
    src_mask = src_mask.repeat(opt.k, 1, 1)
    if opt.device == 0:
        src_mask = src_mask.cuda()
    eos_tok = TRG.vocab.stoi['<eos>']

    ind = None
    for i in range(2, opt.max_strlen):
        trg_mask = nopeak_mask(i, opt)
        trg_mask = trg_mask.repeat(opt.k, 1, 1)

        if opt.use_cond2dec == True:
            output_pep = model.out(model.decoder(outputs[:, :i], e_outputs, cond, src_mask, trg_mask))[:, 5:, :]
        else:
            output_pep = model.out(model.decoder(outputs[:, :i], e_outputs, cond, src_mask, trg_mask))
        out_pep = F.softmax(output_pep, dim=-1)

        outputs, log_scores = k_best_outputs(outputs, out_pep, log_scores, i, opt.k)
        ones = (outputs == eos_tok).nonzero()  # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).cuda()
        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.7
            div = 1 / (sentence_lengths.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    vocab_size = 24
    embedding_dim = 256
    hidden_dim = 512
    output_dim = embedding_dim
    gamma = 1000
    j_nearest = 3

    if opt.previous_beam is not None:
        model = SequenceEmbedder(opt, vocab_size, embedding_dim, hidden_dim, output_dim)
        # gamma=compute_gamma(model, gamma, outputs)
        # print('gamma',gamma)
        beam_similarity = hausdorff_similarity(model, opt.previous_beam, outputs, gamma)
        beam_similarity_threshold = threshold_hausdorff_similarity(model, outputs, gamma, j_nearest)
        # print(beam_similarity, beam_similarity_threshold)
        # 0.00011129726772196591 tensor(2951.5042)
        # if beam_similarity > beam_similarity_threshold:
        #     print('-------------beam_similarity > beam_similarity_threshold')
        #     return None
        # else:
        #     pass
        # print('beam_similarity < beam_similarity_threshold')

        # print(f"Similarity between the previous and current beam: {beam_similarity}")

    opt.previous_beam = outputs.clone()

    if ind is None:
        if len((outputs[0] == eos_tok).nonzero()) == 0:
            np.random.seed(123)
            length = np.random.randint(10, 60)
            # length = 10
        else:
            length = (outputs[0] == eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[0][1:length]])

    else:
        length = (outputs[ind] == eos_tok).nonzero()[0]
        return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])
