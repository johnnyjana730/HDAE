import os
import json
import torch
import random

from os import name
from tqdm import tqdm
from torch import nn, optim
from torch.utils import data
from torch.utils.data import sampler
from sklearn.metrics import f1_score
from datetime import datetime

from Models.get_model import *
from Dataset.get_dataset import *
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
    def __init__(self, hparams) -> None:

        try:
            if not os.path.isdir(hparams.sumout): 
                os.makedirs(hparams.sumout)
        except: pass

        hparams.pic_file = hparams.sumout 
        hparams.sumout_grum_probs = hparams.sumout 

        self.hparams = hparams
        self.asp_cnt = self.hparams.st_num_aspect
        print('loading dataset...')

        # if self.hparams.student_type == 'bert_fin':
        #     self.ds = bt_dataset_label(hparams, hparams.aspect_init_file, hparams.maxlen)
        #     test_ds = bt_test_dataset_label(hparams, hparams.aspect_init_file, hparams.maxlen)

        # else:
        #     self.ds = Dataset_label(hparams, hparams.aspect_init_file, hparams.maxlen)
        #     test_ds = TestDataset_label(hparams, hparams.aspect_init_file, hparams.maxlen)


        self.ds = Dataset(hparams, hparams.aspect_init_file, hparams.maxlen)
        test_ds = TestDataset(hparams, hparams.aspect_init_file, hparams.maxlen)
        # self.ds = Dataset(hparams, hparams.aspect_init_file, hparams.train_file,
        #                   hparams.st_pre, hparams.maxlen)
        # test_ds = TestDataset(
        #     hparams, hparams.aspect_init_file, hparams.test_file)
        self.test_loader = data.DataLoader(test_ds, batch_size=1, num_workers=1)

        print(f'dataset_size: {len(self.ds)}')

        print('loading model...')
        self.idx2asp = torch.tensor(self.ds.get_idx2asp()).cuda()

        self.teacher = Teacher(self.idx2asp, self.asp_cnt,
                               self.hparams.general_asp).cuda()
        self.student = Student(hparams).cuda()

        # for p in  self.student.parameters():
        #     p.requires_grad = True

        params = filter(lambda p: p.requires_grad, self.student.parameters())
        self.student_opt = torch.optim.Adam(params, lr = hparams.lr)

        self.criterion = EntropyLoss().cuda()

        self.z = self.reset_z()

        self.rad_record = {}

        self.tolorence = 3
        self.early_stop = 0
        self.early_stop_criteria = 5

    def save_model(self, path, model_name):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.hparams, f)
        torch.save({'teacher_z': self.z, 'student': self.student.state_dict()}, os.path.join(
            path, f'epoch_{model_name}_student.pt'))

    def train_loader(self, ds):
        return data.DataLoader(ds, batch_size=1, num_workers=1)

    def train(self):
        prev_best = 0
        loader = self.train_loader(self.ds)

        for epoch in range(100):
            self.epoch = epoch
            print(f'Epoch: {epoch}')
            loss = self.train_per_epoch(loader)
            # self.epoch = epoch
            score = self.test(epoch)
            print(f'epoch: {epoch}, f1_mid: {score:.3f}, prev_best: {prev_best:.3f}')

            if prev_best < score:
                self.early_stop = 0
                prev_best = score
            else:
                if epoch >= self.tolorence:
                    self.early_stop += 1
                    if self.early_stop == self.early_stop_criteria: break

    def reset_z(self):
        z = torch.ones(
            (self.asp_cnt, 30)).cuda()
        return z / z.sum(-1).unsqueeze(-1)

    def train_per_epoch(self, loader):

        s_logits_tmp = []
        x_bow_tmp = []
        losses = []
        bt_times = 0
        self.z = self.reset_z()


        for x_bow, x_id, ori in tqdm(loader):

            x_bow, x_id, ori = x_bow[0,:,:], x_id[0,:,:], ori

            # bow, idx, labels, ori = bow[0,:,:,:], idx[0,:,:], labels[0,:,:], ori
            # print('ori = ', ori)
            # print('ori[0][0].decode("utf-8") = ', ori[0][0][0].decode("utf-8"))
            self.keys = ori[0][0][0].decode("utf-8")
            # if self.hparams.training_amont >= random.random()
            if self.keys not in  self.rad_record:
                self.rad_record[self.keys] = random.random()
            # if self.hparams.training_amont >= self.rad_record[keys]:

                # print('index = ', index)
    # 
            x_bow, x_id = x_bow.cuda(), x_id.cuda()
            loss, s_logits, x_bow = self.train_step(x_bow, x_id, ori, bt_times)

            s_logits_tmp.append(s_logits)
            x_bow_tmp.append(x_bow)

            if len(s_logits_tmp) == 1000:
                self.z = self.calc_z(torch.cat(s_logits_tmp, 0), torch.cat(x_bow_tmp, 0))
                s_logits_tmp = []
                x_bow_tmp = []


            bt_times += 1
            # if bt_times >= 100:
            #     break

            try:
                losses.append(loss.item())
            except:
                losses.append(0)
        try:
            losses = sum(losses) / (len(losses) + 1)
            return losses
        except:
            return 0    


    def write_result(self, s_logits, labels, grum_a_probs, ori, file_name = 'trn'):

        ori_tmp = [ori_sent for ori_sent in ori]
        # print('ori = ', ori,' ori_tmp = ', ori_tmp)
        # input()

        grum_a_probs_lb = grum_a_probs.max(-1)[1].detach().cpu().tolist()

        lg = s_logits.max(-1)[1].detach().cpu().tolist()

        file_grum_probs = open(self.hparams.sumout_grum_probs + '/' + file_name + '_disa_grum_{:d}.txt'.format(self.epoch), "a")


        for lg_i in range(len(lg)):
            asps = labels[lg_i].tolist()

            sd_words = self.student.aspect_sds_list[lg[lg_i]]
            sd_words_dis_att = grum_a_probs.detach().cpu()[lg_i, lg[lg_i], :, :].tolist()

            file_grum_probs.write('Sent = ' + ori_tmp[0][lg_i][0].decode("utf-8") + '\n' + 'pre = ' + str(lg[lg_i]) +  ', label = ' \
                 + ','.join([str(as_) for as_ in asps]) \
                 + ', pre, label = ' + str(asps[lg[lg_i]] == 1) + '\n')

            for sd_i in range(30):
                cur_dis_label = str(grum_a_probs_lb[lg_i][lg[lg_i]][sd_i])
                dis_att = [str(round(k, 3)) for k in sd_words_dis_att[sd_i]]

                sd_dis_att_info = "sd = " + sd_words[sd_i] + ' ' * (15 - len(sd_words[sd_i])) + \
                             ", dis label : " + cur_dis_label + ', att : ' + ', '.join(dis_att)  + '\n'
                
                file_grum_probs.write(sd_dis_att_info)

            file_grum_probs.write('\n')

        file_grum_probs.close()

    def write_result_no_lb(self, s_logits, grum_a_probs, ori, file_name = 'trn'):

        ori_tmp = [ori_sent for ori_sent in ori]
        # print('ori = ', ori,' ori_tmp = ', ori_tmp)
        # input()

        grum_a_probs_lb = grum_a_probs.max(-1)[1].detach().cpu().tolist()

        lg = s_logits.max(-1)[1].detach().cpu().tolist()

        file_grum_probs = open(self.hparams.sumout_grum_probs + '/' + file_name + '_disa_grum_{:d}.txt'.format(self.epoch), "a")

        for lg_i in range(len(lg)):

            sd_words = self.student.aspect_sds_list[lg[lg_i]]
            sd_words_dis_att = grum_a_probs.detach().cpu()[lg_i, lg[lg_i], :, :].tolist()

            file_grum_probs.write('Sent = ' + ori_tmp[0][lg_i][0].decode("utf-8") + '\n' + 'pre = ' + str(lg[lg_i]) + '\n')

            for sd_i in range(30):
                cur_dis_label = str(grum_a_probs_lb[lg_i][lg[lg_i]][sd_i])
                dis_att = [str(round(k, 3)) for k in sd_words_dis_att[sd_i]]

                sd_dis_att_info = "sd = " + sd_words[sd_i] + ' ' * (15 - len(sd_words[sd_i])) + \
                             ", dis label : " + cur_dis_label + ', att : ' + ', '.join(dis_att)  + '\n'
                
                file_grum_probs.write(sd_dis_att_info)

            file_grum_probs.write('\n')

        file_grum_probs.close()


    def write_result_2_no_lb(self, s_logits, ori, file_name = 'trn'):

        ori_tmp = [ori_sent for ori_sent in ori]

        lg = s_logits.max(-1)[1].detach().cpu().tolist()

        file_grum_probs = open(self.hparams.sumout_grum_probs + '/' + file_name + '_disa_grum_{:d}.txt'.format(self.epoch), "a")

        for lg_i in range(len(lg)):


            file_grum_probs.write('Sent = ' + ori_tmp[0][lg_i][0].decode("utf-8") + '\n' + 'pre = ' + str(lg[lg_i]) + '\n')

            file_grum_probs.write('\n')

        file_grum_probs.close()

    def write_result_2(self, s_logits, labels, ori, file_name = 'trn'):

        ori_tmp = [ori_sent for ori_sent in ori]

        lg = s_logits.max(-1)[1].detach().cpu().tolist()

        file_grum_probs = open(self.hparams.sumout_grum_probs + '/' + file_name + '_disa_grum_{:d}.txt'.format(self.epoch), "a")

        for lg_i in range(len(lg)):
            asps = labels[lg_i].tolist()

            file_grum_probs.write('Sent = ' + ori_tmp[0][lg_i][0].decode("utf-8") + '\n' + 'pre = ' + str(lg[lg_i]) +  ', label = ' \
                 + ','.join([str(as_) for as_ in asps]) \
                 + ', pre, label = ' + str(asps[lg[lg_i]] == 1) + '\n')

            file_grum_probs.write('\n')

        file_grum_probs.close()

    def train_step(self, x_bow, x_id, x_ori, bt_times):

        if self.hparams.student_type[:len('hyper_rec_dis')] == 'hyper_rec_dis' and bt_times % 20 == 0:
            for g in self.student_opt.param_groups:
                g['lr'] = self.hparams.lr * 2

            for _ in range(10):
                loss_2 = self.student.intra_aspect_model()
                self.student_opt.zero_grad()
                loss_2.backward()
                self.student_opt.step()

            for g in self.student_opt.param_groups:
                g['lr'] = self.hparams.lr


        t_logits = self.teacher(x_bow, self.z)
        loss = 0.
        prev = -1

        self.student_opt.zero_grad()

        if self.hparams.student_type == 'hyper_rec_gbl_fin' or self.hparams.student_type == "hyper_rec_disa_fin_no_lab":

            eu_r, fin_label, _, s_logits, hyper_sent, grum_a_probs = self.student(x_id)
 
            # self.write_result_no_lb(s_logits, grum_a_probs, x_ori, 'trn')

            positives, negatives = self.student.get_targets()

            loss = rec_loss(eu_r, positives, negatives)
            # try:
            aspects = self.student.get_aspects()
            loss += 0.05 * orth_loss(aspects)
            # except:
        
        elif self.hparams.student_type == 'w2v_rec_fin':
            eu_r, _, s_logits = self.student(x_id)

            # self.write_result_2_no_lb(s_logits, x_ori, 'trn')

            positives, negatives = self.student.get_targets()

            loss = rec_loss(eu_r, positives, negatives)
            # try:
            aspects = self.student.get_aspects()
            loss += 0.05 * orth_loss(aspects)
            # except:

        elif  self.hparams.student_type == 'w2v_fin' or self.hparams.student_type == 'bert_fin':

            s_logits = self.student(x_id)


        loss.backward()
        self.student_opt.step()

        tmp = (t_logits.max(-1)[1] == s_logits.max(-1)[1]).sum()
        prev = tmp

        t_logits = self.teacher(x_bow, self.z)

        return loss, s_logits, x_bow

    def test(self, epoch):
        ref = []
        pred = []
        score = []
        result = []
        ground = []
        self.student.eval()


        for batch in self.test_loader:
            bow, idx, labels, ori = batch

            bow, idx, labels, ori = bow[0,:,:,:], idx[0,:,:], labels[0,:,:], ori

            bow = bow.cuda()
            idx = idx.cuda()

            with torch.no_grad():
            
                if self.hparams.student_type == 'hyper_rec_gbl_fin' or self.hparams.student_type == "hyper_rec_disa_fin_no_lab":
                    _, _, _, logits, _, grum_a_probs = self.student(idx)

                    self.write_result(logits, labels, grum_a_probs,  ori, 'tst')

                elif self.hparams.student_type == 'w2v_rec_fin':
                    _, _, logits = self.student(idx)

                    self.write_result_2(logits, labels,  ori, 'tst')

                elif  self.hparams.student_type == 'w2v_fin' or self.hparams.student_type == 'bert_fin':
                    logits = self.student(idx)

                    self.write_result_2(logits, labels,  ori, 'tst')

            lg = logits.max(-1)[1].detach().cpu().tolist()
            
            for lg_i in range(len(lg)):
                asps = labels[lg_i].tolist()

                pred.append(lg[lg_i])
                if asps[lg[lg_i]] == 1:
                    ref.append(lg[lg_i])
                else:
                    for as_i in range(self.hparams.st_num_aspect):
                        if asps[as_i] == 1:
                            ref.append(as_i)
                            break

        score = f1_score(y_true=ref, y_pred=pred, average='micro')

        # eva_file = open(self.hparams.sumout, "a")
        # eva_result = 'f1 score ={}'.format(score)
        # eva_file.write('Epoch = ' + str(epoch) + ' ,' + str(eva_result) + '\n')
        # eva_file.close()
        

        eva_file = open(self.hparams.sumout_2, "a")
        eva_result = 'f1 score ={}'.format(score)
        eva_file.write('Epoch = ' + str(epoch) + ' ,' + str(eva_result) + '\n')
        eva_file.close()

        # exp_info

        self.student.train()
        return score

    def calc_z(self, logits, bow):
        """z
        Args:
            logits: B, asp_cnt
            bow: B, bow_size
        Returns:
            : asp_cnt, bow_size
        """
        val, trg = logits.max(1)
        num_asp = logits.shape[1]

        tmp_asp = []
        for k in range(num_asp):

            if len(bow[torch.where(trg == k)]) != 0:
                true_tmp_asp_z = bow[torch.where(trg == k)][:, k, :].float().sum(0)
            else:
                true_tmp_asp_z = torch.zeros(30).to(bow.device)

            tmp_asp_z = bow[:, k, :].float().sum(0)

            tmp_asp_z = tmp_asp_z.masked_fill(tmp_asp_z == 0., 1e-10)

            tmp_q = true_tmp_asp_z / tmp_asp_z

            tmp_asp.append(tmp_q.unsqueeze(0))

        tmp = torch.cat(tmp_asp, 0)
        tmp = tmp + 0.05
        tmp = tmp / tmp.sum(-1).unsqueeze(-1)

        return tmp

