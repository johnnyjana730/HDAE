import torch
import h5py
import manifolds
import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torch.nn.init import xavier_uniform_


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )

def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


class AttentionEncoder(nn.Module):
    """Segment encoder that produces segment vectors as the weighted
    average of word embeddings.
    """
    def __init__(self, vocab_size, emb_size, bias=True, M=None, b=None):
        """Initializes the encoder using a [vocab_size x emb_size] embedding
        matrix. The encoder learns a matrix M, which may be initialized
        explicitely or randomly.

        Parameters:
            vocab_size (int): the vocabulary size
            emb_size (int): dimensionality of embeddings
            bias (bool): whether or not to use a bias vector
            M (matrix): the attention matrix (None for random)
            b (vector): the attention bias vector (None for random)
        """
        super(AttentionEncoder, self).__init__()
        self.lookup = nn.Embedding(vocab_size, emb_size)
        self.M = nn.Parameter(torch.Tensor(emb_size, emb_size))
        if M is None:
            xavier_uniform_(self.M.data)
        else:
            self.M.data.copy_(M)
        if bias:
            self.b = nn.Parameter(torch.Tensor(1))
            if b is None:
                self.b.data.zero_()
            else:
                self.b.data.copy_(b)
        else:
            self.b = None

    def forward(self, inputs):
        """Forwards an input batch through the encoder"""
        x_wrd = self.lookup(inputs)
        x_avg = x_wrd.mean(dim=1)

        x = x_wrd.matmul(self.M)
        x = x.matmul(x_avg.unsqueeze(1).transpose(1,2))

        a = F.softmax(x, dim=1)

        z = a.transpose(1,2).matmul(x_wrd)
        z = z.squeeze()
        if z.dim() == 1:
            return z.unsqueeze(0)
        return z

    def set_word_embeddings(self, embeddings, fix_w_emb=True):
        """Initialized word embeddings dictionary and defines if it is trainable"""
        self.lookup.weight.data.copy_(embeddings)
        self.lookup.weight.requires_grad = not fix_w_emb


class Student_Hyp_DIS_REC(nn.Module):
    def __init__(self, hparams, neg_samples = 10, bias=True, M=None, b=None) -> None:
        super(Student_Hyp_DIS_REC, self).__init__()


        self.dis_1 = hparams.dis_1
        self.dis_2 = hparams.dis_2
        self.dis_3 = hparams.dis_3
        self.dis_mun = hparams.dis_mun
        self.gb_temp = hparams.gb_temp

        self.manifold = getattr(manifolds, "PoincareBall")()
        self.hparams = hparams
        self.student_init()

        self.seg_encoder = AttentionEncoder(self.hparams.vocab_size, self.hparams.emb_size, bias, M, b)
        self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)
        self.hyper_beta = hparams.hyper_beta
        self.neg_samples = neg_samples
        self.seg_encoder.set_word_embeddings(self.w_emb, False)

        self.num_seeds = self.hparams.num_seeds
        self.num_aspects = self.hparams.num_asp
        self.emb_size = self.hparams.emb_size

    def student_init(self):

        id2word = {}
        word2id = {}
        data_file = './data/preprocessed/' + self.hparams.dataset + '_MATE'
        fvoc = open(data_file + '_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            id2word[int(id)] = word
            word2id[word] = int(id)
        fvoc.close()

        h5py_file = './data/preprocessed/' + self.hparams.dataset + '_MATE'
        f = h5py.File(h5py_file + '.hdf5', 'r')
        
        w_emb_array = f['w2v'][()] * self.hparams.w2v_ratio
        w_emb_array = w_emb_array
        self.w_emb = torch.from_numpy(w_emb_array)
        vocab_size, emb_size = self.w_emb.size()

        self.hparams.vocab_size = vocab_size
        self.hparams.emb_size = emb_size

        self.lookup = nn.Embedding(vocab_size, emb_size)
        self.lookup.weight.data.copy_(self.w_emb)
        self.lookup.weight.requires_grad = True
        
        aspect_seeds_file = self.hparams.aspect_seeds

        fseed = open(aspect_seeds_file, 'r')
        aspects_ids = []
        seed_weights = []

        for line in fseed:
            seeds = []
            weights = []
            for tok in line.split():
                word, weight = tok.split(':')
                if word in word2id:
                    seeds.append(word2id[word])
                    weights.append(float(weight))
                else:
                    seeds.append(0)
                    weights.append(0.0)
            aspects_ids.append(seeds)
            seed_weights.append(weights)

        fseed.close()

        seed_w = torch.Tensor(seed_weights)
        seed_w /= seed_w.norm(p=1, dim=1, keepdim=True)

        clouds = []
        for seeds in aspects_ids:
            clouds.append(w_emb_array[seeds])
        a_emb = torch.from_numpy(np.array(clouds))

        self.a_emb = nn.Parameter(torch.Tensor(a_emb.size()))
        self.a_emb.data.copy_(a_emb)
        self.a_emb.requires_grad = False

        num_asp, num_seeds  = seed_w.size()

        self.hparams.num_seeds = num_seeds
        self.hparams.num_asp = num_asp

        self.seed_w = nn.Parameter(torch.Tensor(seed_w.size()))
        self.seed_w.data.copy_(seed_w)
        self.seed_w.requires_grad = False

        self.lin = nn.Linear(emb_size, self.hparams.st_num_aspect)
        self.softmax = nn.Softmax(dim=1)

        self.softmax_gumbel = nn.Softmax(dim=3)

        dis_a_emb = a_emb.clone()
        dis_a_emb = dis_a_emb.unsqueeze(2).repeat(1, 1, self.dis_mun, 1)
        dis_a_emb = dis_a_emb + torch.randn(dis_a_emb.size()) * 0.00001
        self.a_dis_emb = nn.Parameter(torch.Tensor(dis_a_emb.size()))
        self.a_dis_emb.data.copy_(dis_a_emb)
        self.a_dis_emb.requires_grad = True

    def intra_aspect_model(self):

        a_dis_multi_o2a = self.a_dis_emb.view(self.num_aspects, self.num_seeds * self.dis_mun, -1).unsqueeze(2).repeat(1, 1, self.num_seeds * self.dis_mun, 1)

        a_dis_multi_rpa = self.a_dis_emb.view(self.num_aspects, self.num_seeds * self.dis_mun, -1).repeat(1, self.num_seeds * self.dis_mun, 1)

        hyp_a_dis_multi_o2a = self.manifold.proj_tan0_exp(a_dis_multi_o2a.view(-1, self.emb_size), 1)

        hyp_a_dis_multi_rpa = self.manifold.proj_tan0_exp(a_dis_multi_rpa.view(-1, self.emb_size), 1)

        as_seed_dis_inter_dis = self.manifold.sqdist(hyp_a_dis_multi_o2a, hyp_a_dis_multi_rpa, 1)

        as_seed_dis_inter_dis = as_seed_dis_inter_dis.view(self.num_aspects, self.num_seeds * self.dis_mun, self.num_seeds * self.dis_mun)
        
        multi_weights = torch.ones(self.num_seeds * self.dis_mun, self.num_seeds * self.dis_mun) - block_diag(torch.ones(self.num_seeds, self.dis_mun, self.dis_mun))

        multi_weights = multi_weights.to(as_seed_dis_inter_dis.device)

        inter_as_sd_inter_dis = as_seed_dis_inter_dis * multi_weights.view(1 ,self.num_seeds * self.dis_mun, self.num_seeds * self.dis_mun)

        inter_as_sd_inter_dis_pre = inter_as_sd_inter_dis.view(self.num_aspects, self.num_seeds, self.dis_mun, self.num_seeds, self.dis_mun)

        inter_as_sd_inter_dis, _ = torch.min(inter_as_sd_inter_dis_pre, dim=4)

        inter_as_sd_inter_dis, _ = torch.min(inter_as_sd_inter_dis, dim=2)

        dist_hinge = torch.clamp(inter_as_sd_inter_dis - self.dis_1, min=0.0)

        dist_hinge = dist_hinge.sum(dim=1)

        dist_loss_1 = dist_hinge.sum()

        # part2 intra aspect seeds

        multi_weights_block = block_diag(torch.ones(self.num_seeds, self.dis_mun, self.dis_mun)) - torch.eye(self.num_seeds * self.dis_mun)

        multi_weights_block = multi_weights_block.to(as_seed_dis_inter_dis.device)

        intra_as_sd_intent_dis = (self.dis_2 - as_seed_dis_inter_dis) * multi_weights_block.view(1 ,self.num_seeds * self.dis_mun, self.num_seeds * self.dis_mun)

        intra_as_sd_intent_dis = torch.clamp(intra_as_sd_intent_dis, min=0.0).sum(dim=1)
        # intra aspect intent far

        intra_as_sd_intent_dis = intra_as_sd_intent_dis.sum(-1)
        dist_loss_2 = intra_as_sd_intent_dis.sum(-1)

        # part3 aspect guide aspect seeds vgbhn  

        a_emb_w = self.a_emb.mul(self.seed_w.view(self.num_aspects, self.num_seeds, 1))

        a_emb_w = a_emb_w.sum(dim=1).view(self.num_aspects, 1, 1, self.emb_size)

        repeat_a_emb_w = a_emb_w.repeat(1, self.num_seeds, self.dis_mun, 1)

        hyp_a_dis_emb = self.manifold.proj_tan0_exp(self.a_dis_emb.view(-1, self.emb_size), 1)

        hyp_repeat_a_emb_w = self.manifold.proj_tan0_exp(repeat_a_emb_w.view(-1, self.emb_size), 1)

        g_as_intent_dis = self.manifold.sqdist(hyp_a_dis_emb.view(-1, self.emb_size), hyp_repeat_a_emb_w.view(-1, self.emb_size), 1)

        g_as_intent_dis = torch.clamp(g_as_intent_dis - self.dis_3 , min=0.0)

        dist_loss_3 = g_as_intent_dis.sum()

        return dist_loss_1 + dist_loss_2 + dist_loss_3

    def set_targets(self, module, input, output):
        """Sets positive and negative samples"""
        assert self.cur_mask is not None, 'Tried to set targets without a mask'
        batch_size = output.size(0)

        if torch.cuda.is_available():
            mask = self.cur_mask.cuda()
        else:
            mask = self.cur_mask

        self.negative = output.data.expand(batch_size, batch_size, self.hparams.emb_size).gather(1, mask)
        self.positive = output.data
        self.cur_mask = None

    def aspect_intent_sel(self, enc_aspects):

        bsz, dim = enc_aspects.shape[0], enc_aspects.shape[-1]

        as_sd_den = self.a_dis_emb.view(self.num_aspects, self.num_seeds * self.dis_mun, -1).view(-1, dim)

        hyp_as_sd_den = self.manifold.proj_tan0_exp(as_sd_den.view(-1, dim), 1)

        bz_hyp_as_sd_den = (hyp_as_sd_den.unsqueeze(0)).repeat(bsz,1,1)

        probs_pre = self.manifold.sqdist(enc_aspects.view(-1, dim), bz_hyp_as_sd_den.view(-1, dim), 1)

        probs_pre = probs_pre.view(-1,1)

        probs_exp = torch.exp(- self.hyper_beta * probs_pre - 0.05)

        pre_a_probs = probs_exp.view(bsz, -1)

        pre_a_probs = pre_a_probs.view(bsz, self.num_aspects , self.num_seeds, self.dis_mun)

        pre_a_probs = pre_a_probs / self.gb_temp
        
        grum_a_probs = F.softmax(pre_a_probs, dim=3)

        bz_a_dis_emb = (self.a_dis_emb.unsqueeze(0)).repeat(bsz, 1, 1, 1, 1)

        a_dis_emb_tmp = bz_a_dis_emb.view(bsz, self.num_aspects, self.num_seeds, self.dis_mun, -1) * grum_a_probs.view(bsz, self.num_aspects, self.num_seeds, self.dis_mun, 1)

        a_dis_emb_tmp = a_dis_emb_tmp.sum(dim=3)

        return a_dis_emb_tmp

    def forward(self, inputs, batch_num=None):
        if self.training:
            # mask used for randomly selected negative examples
            self.cur_mask = self._create_neg_mask(inputs.size(0))

        enc = self.seg_encoder(inputs)

        bsz, dim = enc.shape[0], enc.shape[1]

        enc = self.manifold.proj_tan0_exp(enc.view(-1, dim), 1)

        hyper_sent = enc

        enc_aspects = (enc.unsqueeze(1)).repeat(1, self.num_aspects * self.num_seeds * self.dis_mun, 1)

        a_dis_emb_tmp = self.aspect_intent_sel(enc_aspects)

        bz_a_emb_w = a_dis_emb_tmp.mul(self.seed_w.view(1, self.num_aspects, self.num_seeds, 1))
        
        bz_a_emb_w = bz_a_emb_w.sum(dim=2)

        enc_aspects_2 = (enc.unsqueeze(1)).repeat(1,self.num_aspects,1)

        hyp_a_w = self.manifold.proj_tan0_exp(bz_a_emb_w.view(-1, dim), 1)

        probs_pre = self.manifold.sqdist(enc_aspects_2.view(-1, dim), hyp_a_w.view(-1, dim), 1)

        probs_pre = probs_pre.view(-1,1)

        probs_exp = torch.exp(- self.hyper_beta * probs_pre - 0.05)

        kle_a_w_emb = self.manifold.to_klein(hyp_a_w.view(-1, dim), 1)

        probs_exp = probs_exp.view(bsz, self.num_aspects)

        a_probs = (probs_exp / (probs_exp.sum(1).unsqueeze(1))).view(bsz, self.num_aspects)

        probs = self.manifold.lorentz_factor(kle_a_w_emb, 1).view(bsz, self.num_aspects, 1) * probs_exp.view(bsz, self.num_aspects, 1)

        probs = probs.view(-1, self.num_aspects)

        probs_expanded = (probs / (probs.sum(1).unsqueeze(1))).unsqueeze(-1)
        
        if self.manifold.name == 'Hyperboloid':
            kle_a_w_emb = kle_a_w_emb.view(bsz, self.num_aspects, dim-1)
        else:
            kle_a_w_emb = kle_a_w_emb.view(bsz, self.num_aspects, dim)

        kle_o = ((kle_a_w_emb * probs_expanded).sum(dim=1)).unsqueeze(1)

        if self.manifold.name == 'Hyperboloid':
            r = self.manifold.klein_to(kle_o.view(-1, dim-1), 1)
        else:
            r = self.manifold.klein_to(kle_o.view(-1, dim), 1)

        r = r.squeeze()

        eu_r = self.manifold.logmap0(r.view(-1, r.shape[-1]), 1.0)

        return eu_r, probs_expanded.squeeze(1), a_probs.squeeze(1), hyper_sent

    def _create_neg_mask(self, batch_size):
        """Creates a mask for randomly selecting negative samples"""
        multi_weights = torch.ones(batch_size, batch_size) - torch.eye(batch_size)
        neg = min(batch_size - 1, self.neg_samples)

        mask = torch.multinomial(multi_weights, neg)

        mask = mask.unsqueeze(2).expand(batch_size, neg, self.hparams.emb_size)

        mask = Variable(mask, requires_grad=False)
        return mask

    def get_targets(self):
        assert self.positive is not None, 'Positive targets not set; needs a forward pass first'
        assert self.negative is not None, 'Negative targets not set; needs a forward pass first'
        return self.positive, self.negative

    def get_aspects(self):
        if self.a_emb.dim() == 2:
            return self.a_emb
        else:
            return self.a_emb.mean(dim=1)

    def train(self, mode=True):
        super(Student_Hyp_DIS_REC,  self).train(mode)
        if self.encoder_hook is None:
            self.encoder_hook = self.seg_encoder.register_forward_hook(self.set_targets)
        return self

    def eval(self):
        super(Student_Hyp_DIS_REC, self).eval()
        if self.encoder_hook is not None:
            self.encoder_hook.remove()
            self.encoder_hook = None
        return self