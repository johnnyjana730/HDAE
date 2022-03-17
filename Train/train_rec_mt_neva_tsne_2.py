import os
import json
import torch

from torch import nn
from tqdm import tqdm
from torch.utils import data
from sklearn.metrics import f1_score

from Models.get_model import *
from Dataset.get_dataset import *
from Train.plot import plot_figure
from Train.t_sne import tsne_embedding
from loss import TripletMarginCosineLoss, OrthogonalityLoss, TripletMarginCosineLoss_hyperdis

torch.backends.cudnn.benchmark = True
torch.set_printoptions(profile="full")

orth_loss = OrthogonalityLoss()
rec_loss = TripletMarginCosineLoss()
rec_loss_dis = TripletMarginCosineLoss_hyperdis()

class EntropyLoss(nn.Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, p, q):
        b = q * torch.log(p)
        b = -1. * b.sum()
        return b

class Trainer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.tolerance = 3
        self.early_stop = 0
        self.early_stop_criterion = 5
        self.num_aspect = self.hparams.st_num_aspect

        train_data = Dataset(hparams, hparams.aspect_init_file, hparams.maxlen)
        test_data = TestDataset(hparams, hparams.aspect_init_file, hparams.maxlen)

        self.train_loader = data.DataLoader(train_data, batch_size=1, num_workers=1)
        self.test_loader = data.DataLoader(test_data, batch_size=1, num_workers=1)

        self.teacher = Teacher(self.num_aspect, self.hparams.general_asp).cuda()
        self.student = Student(hparams).cuda()
        self.student_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.student.parameters()),
            lr = hparams.lr
        )
        self.criterion = EntropyLoss().cuda()
        self.z = self.reset_z()

    def save_model(self, path, model_name):
        if not os.path.exists(path):
            os.mkdir(path)

        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.hparams, f)

        torch.save(
            {'teacher_z': self.z, 'student': self.student.state_dict()},
            os.path.join(path, f'epoch_{model_name}_student.pt')
        )

    def train(self):
        prev_best = 0

        for epoch in range(30):
            self.curr_epoch = epoch

            self.total_sentence_vec = list()
            loss = self.train_per_epoch(self.train_loader)
            f_score = self.test_per_epoch(self.test_loader)

            if prev_best < f_score:
                self.early_stop = 0
                prev_best = f_score
            elif epoch >= self.tolerance:
                self.early_stop += 1
                if self.early_stop == self.early_stop_criterion: break

    def reset_z(self):
        z = torch.ones((self.num_aspect, 30)).cuda()
        return z / z.sum(-1).unsqueeze(-1)

    def train_per_epoch(self, loader):
        self.z = self.reset_z()

        losses = list()
        s_logits_list, x_bow_list = [], []

        for i, (x_bow, x_id, ori) in tqdm(enumerate(loader)):
            x_bow, x_id = x_bow[0, :, :], x_id[0, :, :]
            x_bow, x_id = x_bow.cuda(), x_id.cuda()
            loss, s_logits, x_bow = self.train_step(x_bow, x_id, ori, i)

            s_logits_list.append(s_logits)
            x_bow_list.append(x_bow)

            if len(s_logits_list) == 1000:
                self.z = self.calc_z(torch.cat(s_logits_list, 0), torch.cat(x_bow_list, 0))
                s_logits_list, x_bow_list = [], []

            losses.append(loss.item())

        losses = sum(losses) / len(losses)
        return losses
    
    def train_step(self, x_bow, x_id, x_ori, step_i):
        if self.hparams.student_type[:len('hyper_rec_dis')] == 'hyper_rec_dis' and step_i % 20 == 0:
            for group in self.student_optim.param_groups:
                group['lr'] = self.hparams.lr * 2

            for _ in range(10):
                self.student_optim.zero_grad()

                loss = self.student.intra_aspect_model()
                loss.backward()
                self.student_optim.step()

            for group in self.student_optim.param_groups:
                group['lr'] = self.hparams.lr

        t_logits = self.teacher(x_bow, self.z)

        self.student_optim.zero_grad()

        eu_r, _, s_logits, hyper_sent = self.student(x_id)
        positives, negatives = self.student.get_targets()
        loss = rec_loss(eu_r, positives, negatives)
        loss += self.hparams.mt_ratio * self.criterion(s_logits, t_logits)
        loss += 0.05 * orth_loss(self.student.get_aspects())

        loss.backward()
        self.student_optim.step()

        hyper_sent =  hyper_sent.detach().cpu().tolist()
        if step_i >= self.hparams.aspect_tsne_bt:
            self.total_sentence_vec = self.total_sentence_vec[len(hyper_sent):]
        self.total_sentence_vec.extend(hyper_sent)

        t_logits = self.teacher(x_bow, self.z)

        return loss, s_logits, x_bow

    def test_per_epoch(self, loader):
        self.student.eval()

        y_true, y_pred = list(), list()

        true_label = {i: 0 for i in range(self.hparams.st_num_aspect)}
        true_predict = {i: 0 for i in range(self.hparams.st_num_aspect)}
        total_predict = {i: 0 for i in range(self.hparams.st_num_aspect)}

        vec_type_set = {i: {} for i in range(self.hparams.st_num_aspect)}

        for bow, idx, labels, ori in loader:
            bow, idx, labels = bow[0,: , :, :], idx[0, :, :], labels[0, :, :]
            bow, idx = bow.cuda(), idx.cuda()

            with torch.no_grad():
                _, _, logits, hyper_sent = self.student(idx)

            hyper_sent =  hyper_sent.detach().cpu().tolist()
            self.total_sentence_vec.extend(hyper_sent)

            max_logit_indices = logits.max(-1)[1].detach().cpu().tolist()
            y_pred.extend(max_logit_indices)
            
            for i in range(len(max_logit_indices)):
                sample_aspects = labels[i].tolist()

                vec_type_set[sample_aspects.index(1)][len(self.total_sentence_vec)-len(hyper_sent)+i] = 1

                total_predict[max_logit_indices[i]] += 1

                if sample_aspects[max_logit_indices[i]] == 1:
                    y_true.append(max_logit_indices[i])

                    true_predict[max_logit_indices[i]] += 1
                    true_label[max_logit_indices[i]] += 1
                else:
                    y_true.append(sample_aspects.index(1))
                    true_label[sample_aspects.index(1)] += 1

        f_score = f1_score(y_true=y_true, y_pred=y_pred, average='micro')

        with open(self.hparams.sumout, 'a') as fout:
            fout.write(f'Epoch = {self.curr_epoch}, F1 score = {f_score}\n\n')

        tsne_embedding(self.total_sentence_vec, vec_type_set, self.curr_epoch)
        plot_figure(true_predict, total_predict, true_label, self.curr_epoch)
        
        self.student.train()

        return f_score

    def calc_z(self, logits, bow):
        """z
        Args:
            logits: B, num_aspect
            bow: B, bow_size
        Returns:
            : num_aspect, bow_size
        """

        z = list()

        max_logit_indices = logits.max(1)[1]
        for k in range(self.num_aspect):
            if len(bow[torch.where(max_logit_indices == k)]) != 0:
                true_aspect_z = bow[torch.where(max_logit_indices == k)][:, k, :].float().sum(0)
            else:
                true_aspect_z = torch.zeros(30).to(bow.device)

            all_aspect_z = bow[:, k, :].float().sum(0)
            all_aspect_z = all_aspect_z.masked_fill(all_aspect_z == 0., 1e-10)

            aspect_z = true_aspect_z / all_aspect_z
            z.append(aspect_z.unsqueeze(0))

        z = torch.cat(z, 0)
        z += 0.05
        z = z / z.sum(-1).unsqueeze(-1)
        return z