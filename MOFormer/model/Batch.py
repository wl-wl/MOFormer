import torch
from torchtext import data
import numpy as np
from torch.autograd import Variable


def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    if opt.use_cond2dec == True:
        cond_mask = np.zeros((1, opt.cond_dim, opt.cond_dim))
        cond_mask_upperright = np.ones((1, opt.cond_dim, size))
        cond_mask_upperright[:, :, 0] = 0
        cond_mask_lowerleft = np.zeros((1, size, opt.cond_dim))
        upper_mask = np.concatenate([cond_mask, cond_mask_upperright], axis=2)
        lower_mask = np.concatenate([cond_mask_lowerleft, np_mask], axis=2)
        np_mask = np.concatenate([upper_mask, lower_mask], axis=1)
    np_mask = Variable(torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
      np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, cond, opt):
    torch.set_printoptions(profile="full")
    src_mask = (src != opt.src_pad).unsqueeze(-2)
    cond_mask = torch.unsqueeze(cond, -2)
    cond_mask = torch.ones_like(cond_mask, dtype=bool)
    src_mask = torch.cat([cond_mask, src_mask], dim=2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        if opt.use_cond2dec == True:
            trg_mask = torch.cat([cond_mask, trg_mask], dim=2)
        np_mask = nopeak_mask(trg.size(1), opt)
        if trg.is_cuda:
            np_mask.cuda()
        trg_mask = trg_mask.to(np_mask.device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask


class MyIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

global max_src_in_batch, max_tgt_in_batch

def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)
