# -*- coding: utf-8 -*-
# import-module
import os
import csv
import matplotlib.pyplot as plt

import jieba.posseg as pseg
import nengo
import numpy as np
from nengo.exceptions import SpaParseError, ValidationError

from tensorflow.python.keras.models import load_model

import pickle
import rospkg
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

NODE_NAME = 'text_analysis'
rospack = rospkg.RosPack()
path = rospack.get_path(NODE_NAME)
SP_MODEL = '/model/sp_30epochs_500dim.h5'
SP_VOCAB = '/model/vocab.pickle'
DIM = 500

def segmentation_postagger(train_data):
    train_data_seg = []
    for i in train_data:
        train_data_seg.append(pseg.lcut(i))
    return train_data_seg

class online_classification:

    def __init__(self):
        self.model = load_model(path+SP_MODEL)
        self.load_vocab(path+SP_VOCAB)
    # Use pretrained semantic pointer vocab
    def load_vocab(self, vocab_name):
        with open(vocab_name, 'rb') as f:
            self.vocab = pickle.load(f)

    def sentence2sp(self, sentence_seg_pos):
        sentence_sp = None
        new_s = self.vocab['Start']
        for word, flag in sentence_seg_pos:
            new_token = self.vocab['V' + word.encode('utf-8')] * self.vocab[flag.upper()]
            new_s += new_token
        new_s.normalize()  # normalize with norm 2.
        # sentence_sp = new_s.v/(len(sentence_seg_pos)+1)# basic normalize method
        sentence_sp = new_s.v
        return sentence_sp

    # Online demo, predict keyboard input sentences
    def online_demo(self):
        while(True):
            sentence = []
            s = input('say something ... ')
            if s == 'bye':
                break
            sentence.append(s)
            sentence = np.array(sentence)
            sentence_seg_pos = segmentation_postagger(sentence)
            sp = np.zeros((1, DIM))
            sp[0] = self.sentence2sp(sentence_seg_pos[0])
            predict = self.model.predict(sp)
            predict_cls = np.round(predict)

    def normal_analysis(self, sentence_in):
        sentence = []
        sentence.append(sentence_in)
        sentence = np.array(sentence)
        sentence_seg_pos = segmentation_postagger(sentence)
        sp = np.zeros((1, DIM))
        sp[0] = self.sentence2sp(sentence_seg_pos[0])
        predict = self.model.predict(sp)
        predict_cls = np.round(predict)
        if predict_cls==1:
            return 'abnomal'
        else:
            return 'normal'

if __name__ == '__main__':
    test_online = online_classification()
    test_online.online_demo()
