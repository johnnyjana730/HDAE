import h5py
import torch
import itertools
import numpy as np
from random import shuffle
from torch.utils import data
from torch.autograd import Variable
from sklearn.feature_extraction.text import CountVectorizer


def shuffle_lists(*lists):
    l = list(zip(*lists))
    shuffle(l)
    return zip(*l)


class Dataset(data.Dataset):
    def __init__(self, hparams, aspect_init_file, maxlen=10):
        self.maxlen = maxlen
        self.hparams = hparams
        self.aspects, vocab = self.load_aspect_seed_words(aspect_init_file)

        self.vector_list = []
        for seeds in self.aspects:
            seeds = list(set(seeds))
            while len(seeds) < 30: seeds.append(seeds[-1] + '-')

            cv = CountVectorizer(vocabulary=sorted(seeds))
            cv.fixed_vocabulary_ = True
            self.vector_list.append(cv)

        self.vectorizer = CountVectorizer(vocabulary=sorted(list(set(vocab))))
        self.vectorizer.fixed_vocabulary_ = True

        self.init_train()

    def init_train(self):
        self.id2word = {}
        self.word2id = {}

        fvoc = open('./data/preprocessed/' + self.hparams.dataset + '_MATE_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            self.id2word[int(id)] = word
            self.word2id[word] = int(id)
        fvoc.close()

        self.batches, self.original, self.scodes = [], [], []
        f = h5py.File('./data/preprocessed/' + self.hparams.dataset + '_MATE' + '.hdf5', 'r')
        for b in f['data']:
            if Variable(torch.from_numpy(f['data/' +  b][()]).long()).shape[0] == 1: continue
            self.batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            self.original.append(list(f['original/' + b][()]))
            self.scodes.append(list(f['scodes/' + b][()]))
        f.close()

        self.batches, self.original, self.scodes = shuffle_lists(self.batches, self.original, self.scodes)

    @staticmethod
    def load_aspect_seed_words(file):
        with open(file) as fin:
            seed_words = fin.read().strip().split('\n')
            seed_words = [words.strip().split() for words in seed_words]
            flatten_seed_words = list(itertools.chain(*seed_words))
        return seed_words, flatten_seed_words

    def __getitem__(self, index: int):
        bows = list()
        for ids in self.batches[index]:
            sentence = ' '.join(self.id2word[int(id)] for id in ids)
            bows.append([cv.transform([sentence]).toarray()[0] for cv in self.vector_list])
        return torch.from_numpy(np.array(bows)), torch.LongTensor(self.batches[index]), [self.original[index]]

    def __len__(self):
        return len(self.batches)


class LabelDataset(Dataset):
    def __init__(self, hparams, aspect_init_file, maxlen=10):
        super(LabelDataset, self).__init__(hparams, aspect_init_file, maxlen)

    def init_train(self):
        self.id2word = {}
        self.word2id = {}

        fvoc = open('./data/preprocessed/' + self.hparams.dataset + '_MATE_label_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            self.id2word[int(id)] = word
            self.word2id[word] = int(id)
        fvoc.close()

        self.batches, self.labels, self.original, self.scodes = [], [], [], []
        f = h5py.File('./data/preprocessed/' + self.hparams.dataset + '_MATE_label' + '.hdf5', 'r')
        for b in f['data']:
            if Variable(torch.from_numpy(f['data/' +  b][()]).long()).shape[0] == 1: continue
            self.batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            self.labels.append(Variable(torch.from_numpy(f['labels/' +  b][()]).long()))

            self.original.append(list(f['original/' + b][()]))
            self.scodes.append(list(f['scodes/' + b][()]))
        f.close()

        self.batches, self.labels, self.original, self.scodes = shuffle_lists(self.batches, self.labels, self.original, self.scodes)

    def __getitem__(self, index: int):
        bows = list()
        for ids in self.batches[index]:
            sentence = ' '.join(self.id2word[int(id)] for id in ids)
            bows.append([cv.transform([sentence]).toarray()[0] for cv in self.vector_list])
        return torch.from_numpy(np.array(bows)), torch.LongTensor(self.batches[index]), self.labels[index], [self.original[index]], index


class TestDataset(Dataset):
    def __init__(self, hparams, aspect_init_file, maxlen=10):
        super(TestDataset, self).__init__(hparams, aspect_init_file, maxlen)

    def init_train(self):
        self.id2word = {}
        self.word2id = {}

        fvoc = open('./data/preprocessed/' + self.hparams.dataset + '_MATE_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            self.id2word[int(id)] = word
            self.word2id[word] = int(id)
        fvoc.close()

        self.batches, self.labels, self.original, self.scodes = [], [], [], []
        f = h5py.File("./data/preprocessed/" + self.hparams.dataset + "_MATE_TEST" + '.hdf5', 'r')
        for b in f['data']:
            self.batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            self.labels.append(Variable(torch.from_numpy(f['labels/' +  b][()]).long()))
            self.original.append(list(f['original/' + b][()]))
            self.scodes.append(list(f['scodes/' + b][()]))
        f.close()

    def __getitem__(self, index: int):
        bows = []
        for ids in self.batches[index]:
            sentence = ' '.join(self.id2word[int(id)] for id in ids)
            bows.append([cv.transform([sentence]).toarray()[0] for cv in self.vector_list])
        return torch.from_numpy(np.array(bows)), torch.LongTensor(self.batches[index]), self.labels[index], [self.original[index]]


class TestLabelDataset(Dataset):
    def __init__(self, hparams, aspect_init_file, maxlen=10):
        super(TestLabelDataset, self).__init__(hparams, aspect_init_file, maxlen)

    def init_train(self):
        self.id2word = {}
        self.word2id = {}

        fvoc = open('./data/preprocessed/' + self.hparams.dataset + '_MATE_label_word_mapping.txt', 'r')
        for line in fvoc:
            word, id = line.split()
            self.id2word[int(id)] = word
            self.word2id[word] = int(id)
        fvoc.close()

        self.batches, self.labels, self.original, self.scodes = [], [], [], []
        f = h5py.File("./data/preprocessed/" + self.hparams.dataset + "_MATE_label_TEST" + '.hdf5', 'r')
        for b in f['data']:
            self.batches.append(Variable(torch.from_numpy(f['data/' +  b][()]).long()))
            self.labels.append(Variable(torch.from_numpy(f['labels/' +  b][()]).long()))
            self.original.append(list(f['original/' + b][()]))
            self.scodes.append(list(f['scodes/' + b][()]))
        f.close()

    def __getitem__(self, index: int):
        bows = list()
        for ids in self.batches[index]:
            sentence = ' '.join(self.id2word[int(id)] for id in ids)
            bows.append([cv.transform([sentence]).toarray()[0] for cv in self.vector_list])
        return torch.from_numpy(np.array(bows)), torch.LongTensor(self.batches[index]), self.labels[index], [self.original[index]]