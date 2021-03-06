import numpy as np
from collections import OrderedDict

import utils
import lstm_ctc_model


# data
# mfccs
path_utterances = '/home/deniz/Schreibtisch/ctcLstmNN/mfccFeatures13Pickled'
# alignments as tuples (phone_id, frames)
path_alignments= '/home/deniz/Schreibtisch/ctcLstmNN/rikoSkripts/phnrec/alis.txt'
# phones
path_phones = '/home/deniz/Schreibtisch/ctcLstmNN/rikoSkripts/phnrec/phones.txt'

utterance_dict = utils.load_pickled_file(path_utterances)

phones = utils.load_phone_file(path_phones)

int2sym = utils.make_int2sym(phones)

sym2int = utils.make_sym2int(int2sym)

min_phone, max_phone = utils.min_max_int2sym(int2sym)

alignments = utils.filter_alignments(path_alignments, 50, min_phone, max_phone)

mfcc_chunks, alignment_chunks = utils.make_chunks(utterance_dict, alignments)

inputs = OrderedDict()
for utt_key, mfcc in mfcc_chunks:
    inputs[utt_key] = mfcc

targets = OrderedDict()
for ali_key, ali in alignment_chunks:
    targets[ali_key] = ali

t_inputs = list()
t_targets = list()

for utt_key, mfcc in inputs.items():
    t_inputs.append(mfcc)
    t_targets.append(np.asarray(targets[utt_key]))

train_input = np.asarray(t_inputs)
train_inputs_set = train_input[:100]
train_inputs_val_set = train_input[101:111]

train_target = np.asarray(t_targets)
train_targets_set = train_target[:100]
train_targets_val_set = train_target[101:111]

feature_size = train_input[0].shape[1]
num_classes = len(int2sym) + 1

num_epochs = 1024
num_hidden = 40
num_layers = 1
batch_size = 32
learning_rate = 0.0001
momentum = 0.9

num_train_examples = 100
num_val_examples = 10
num_batches_per_epoch = int(num_train_examples/batch_size)

lstm_ctc_model.run_model(train_inputs_set, train_targets_set, train_inputs_val_set, train_targets_val_set, feature_size,
            num_train_examples, num_val_examples, num_epochs, batch_size, num_batches_per_epoch, learning_rate, momentum,
            num_layers, num_hidden, num_classes)
