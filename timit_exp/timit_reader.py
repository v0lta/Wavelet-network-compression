# Created by moritz (wolter@cs.uni-bonn.de) at 04/12/2019
import kaldiio


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
            key = line.split(' ')[0]
            text = line.split(' ')[1:]
            self.train_text_dict[key] = text

        self.phoneme_map_file = open(self.timit_path + 'data/lang/phones.txt', 'r')
        self.phoneme_dict = {}
        for phoneme_key_str in self.phoneme_map_file.readlines():
            phoneme = phoneme_key_str.split(' ')[0]
            key = int(phoneme_key_str.split(' ')[1][:-1])
            self.phoneme_dict[key] = phoneme
        self.inv_phoneme_dict = {v: k for k, v in self.phoneme_dict.items()}

    def get_train_epoch_dict(self):
        # TODO: pad to max length. Get one hot encoded vectors. Prepeare nice batches.
        train_dict = {}
        for key in self.train_feats_dict:
            print(key)

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