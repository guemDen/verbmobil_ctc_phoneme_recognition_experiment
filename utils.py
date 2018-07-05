import sys
import pickle
import numpy as np

from operator import itemgetter


def load_pickled_file(filename):

    in_file = open(filename, 'rb')
    l_dict = pickle.load(in_file, encoding='latin1')
    in_file.close()

    return l_dict


# latest phones file still have <eps> and #1, etc. symbols in
def load_phone_file(filename):

    phones = list()
    with open(filename) as f:
        for line in f:
            p, _ = line.strip().split()
            if p == '<eps>':
                continue
            if p.startswith('#'):
                break
            phones.append(p)

    return phones


def make_int2sym(phones):

    int2sym = {k: v for k, v in enumerate(phones)}

    return int2sym


def make_sym2int(int2sym):

    sym2int = {v: k for k, v in int2sym.items()}

    return sym2int


def min_max_int2sym(int2sym):

    min_phone = min(int2sym.keys())
    max_phone = max(int2sym.keys())

    return min_phone, max_phone


def filter_alignments(filename, alignment_length, min_phone, max_phone):

    alignments = list()
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            utterance, alignment = line.strip().split(' ', 1)
            alignment = list(map(int, alignment.replace(' ;', '').split()))
            alignment_class_labels_mapped_down_by_1 = [(i-1) for i in alignment[::2]]
            alignment = list(zip(alignment_class_labels_mapped_down_by_1, alignment[1::2]))
            if len(alignment) < alignment_length:
                print("Ignoring %d: %s sequence its too short (%d)" % (i, utterance, len(alignment)), file=sys.stderr)
            if min(map(itemgetter(0), alignment)) < min_phone or max(map(itemgetter(0), alignment)) > max_phone:
                print("Ignoring invalid alignment in line %d: %s" % (i, line), file=sys.stderr)
            alignments.append((utterance, alignment))

    return alignments


def make_chunks(utterance_dict, alignments, stride=1, alignment_length=50):

    alignment_chunks = list()
    mfcc_chunks = list()
    egs = list()
    for utterance_key, alignment in alignments:
        i = 0
        offset = 0
        while i < len(alignment) - alignment_length:
            # chunking alignments
            outs = alignment[i:i+alignment_length]
            labels = list(map(itemgetter(0), outs))
            egs_length = sum(map(itemgetter(1), outs))
            egs.append((utterance_key+'_'+format(i, '04d'), labels, offset, offset+egs_length))
            alignment_chunks.append((utterance_key+'_'+format(i, '04d'), labels))
            # chunking mfccs
            _, _, begin_mfcc, end_mfcc = egs[i]
            mfcc = utterance_dict[utterance_key]
            mfcc_chunk = mfcc[begin_mfcc:end_mfcc]
            mfcc_chunks.append((utterance_key+'_'+format(i, '04d'), mfcc_chunk))

            labels_to_skip = alignment[i:i+stride]
            frames_to_skip = sum(map(itemgetter(1), labels_to_skip))
            offset += frames_to_skip
            i += stride

    return mfcc_chunks, alignment_chunks


def pad_sequences(sequences, maxlen=None, dtype=np.float32, padding='post', truncating='post', value=0.):

   lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

   nb_samples = len(sequences)
   if maxlen is None:
       maxlen = np.max(lengths)

   sample_shape = tuple()
   for s in sequences:
       if len(s) > 0:
           sample_shape = np.asarray(s).shape[1:]
           break

   x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
   for idx, s in enumerate(sequences):
      if len(s) == 0:
         continue
      if truncating == 'pre':
         trunc = s[-maxlen:]
      elif truncating == 'post':
         trunc = s[:maxlen]
      else:
         raise ValueError('Truncating type "%s" not understood' % truncating)

      trunc = np.asarray(trunc, dtype=dtype)
      if trunc.shape[1:] != sample_shape:
         raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' % (trunc.shape[1:], idx, sampleShape))

      if padding == 'post':
         x[idx, :len(trunc)] = trunc
      elif padding == 'pre':
         x[idx, -len(trunc):] = trunc
      else:
         raise ValueError('Padding type "%s" not understood' % padding)

   return x, lengths


def sparse_tuple_from(sequences, dtype=np.int32):

   indices = []
   values = []

   for n, seq in enumerate(sequences):
      indices.extend(zip([n]*len(seq), range(len(seq))))
      values.extend(seq)

   indices = np.asarray(indices, dtype=np.int64)
   values = np.asarray(values, dtype=dtype)
   shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

   return indices, values, shape


def splice(utterance, contextWidth):

   if utterance.shape[0] < 1+2*contextWidth:
      return None

   utt_spliced = np.zeros(
       shape=[utterance.shape[0], utterance.shape[1]*(1+2*contextWidth)],
       dtype=np.float32)

   # middle part
   utt_spliced[:, contextWidth * utterance.shape[1]:
                  (contextWidth + 1) * utterance.shape[1]] = utterance

   for i in range(contextWidth):
       # left context
       utt_spliced[i+1:utt_spliced.shape[0], (contextWidth-i-1)*utterance.shape[1]:(contextWidth-i)*utterance.shape[1]] = utterance[0:utterance.shape[0]-i-1, :]

       # right context
       utt_spliced[0:utt_spliced.shape[0]-i-1, (contextWidth+i+1)*utterance.shape[1]:(contextWidth+i+2)*utterance.shape[1]] = utterance[i+1:utterance.shape[0], :]

   return utt_spliced