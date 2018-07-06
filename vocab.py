from collections import namedtuple

Vocab = namedtuple('Vocab', ['x', 'y', 'W', 'b', 'seq_len', 'logits', 'cost', 'optimizer', 'label_error_rate'])
vocab = Vocab('x', 'y', 'W', 'b', 'seq_len', 'logits', 'cost', 'optimizer', 'label_error_rate')