# Created by moritz (wolter@cs.uni-bonn.de) at 04/12/2019
import kaldiio
import numpy as np
import torch
import random


class TIMITDataSet(object):
    def __init__(self, timit_path='/home/moritz/uni/wavelet_learning/data_sets/timit_folder/kaldi/egs/timit/s5/',
                 batch_size=50):
        self.timit_path = timit_path
        self.timit_train_path = timit_path + 'data/train/'
        self.train_feats_dict = kaldiio.load_scp(self.timit_train_path + 'feats.scp')
        self.batch_size = batch_size
        self.train_text = open(self.timit_train_path + 'text')
        # convert to dict
        self.train_text_dict = {}
        for line in self.train_text.readlines():
            line = line.rstrip('\r\n')
            key = line.split(' ')[0]
            text = line.split(' ')[1:]
            # replace the sil\n with sil
            self.train_text_dict[key] = text

        self.phoneme_map_file = open(self.timit_path + 'data/lang/phones.txt', 'r')
        self.phoneme_dict = {}
        for phoneme_key_str in self.phoneme_map_file.readlines():
            phoneme = phoneme_key_str.split(' ')[0]
            key = int(phoneme_key_str.split(' ')[1][:-1])
            self.phoneme_dict[key] = phoneme
        # self.phoneme_dict[51] = 'pad'
        self.inv_phoneme_dict = {v: k for k, v in self.phoneme_dict.items()}

        self.train_keys, self.train_feats, self.train_phones, self.train_feat_len_lst, self.train_phone_len_lst \
            = self.load_train_list()

    def one_hot_encodings(self, phoneme_lst):
        """
        Given the text produce one hot encoded target vectors.
        Returns: matrix of shape [time, dict_size].
        """
        vec_lst = []
        for phoneme in phoneme_lst:
            one_hot = np.zeros([len(self.phoneme_dict)])
            one_hot[self.inv_phoneme_dict[phoneme]] = 1
            vec_lst.append(one_hot)
        return np.stack(vec_lst)

    def load_train_list(self):
        train_feat_lst = []
        train_feat_len_lst = []
        train_phone_lst = []
        train_phone_len_lst = []
        key_lst = []
        for key in self.train_feats_dict:
            key_lst.append(key)
            feat = torch.from_numpy(self.train_feats_dict[key].astype(np.float32)).cuda()
            phone = torch.from_numpy(self.one_hot_encodings(self.train_text_dict[key]).astype(np.float32)).cuda()
            train_feat_lst.append(feat)
            train_feat_len_lst.append(feat.shape[0])
            train_phone_lst.append(phone)
            train_phone_len_lst.append(phone.shape[0])
        return key_lst, train_feat_lst, train_phone_lst, train_feat_len_lst, train_phone_len_lst

    def get_train_batches(self):
        """
        Returns a list of sequence keys, features and phoneme list.
        """
        combined_lst = list(zip(self.train_keys, self.train_feats, self.train_phones,
                                self.train_feat_len_lst, self.train_phone_len_lst))
        random.shuffle(combined_lst)
        keys, feats, phones, feat_len, phone_len = zip(*combined_lst)
        return keys, feats, phones, feat_len, phone_len

    def get_dev_batches(self):
        pass

    def get_test_batches(self):
        pass

    def get_max_length(self):
        max_len = 0
        for key in self.train_feats_dict:
            len = self.train_feats_dict[key].shape[0]
            if len > max_len:
                max_len = len
                print(key, len)



if __name__ == "__main__":
    print('testing the reader.')
    print('/home/moritz/uni/wavelet_learning/data_sets/timit_folder/kaldi/egs/timit/s5/data/train')
    timit_data = TIMITDataSet()
    # print(len(timit_data.train_feats))
    timit_data.get_max_length()
    timit_data.get_train_batches()