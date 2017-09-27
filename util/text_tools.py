import numpy as np


class BatchGenerator(object):
    """Generate and hold batches."""
    def __init__(self, text, batch_size, n_unrollings, vocab_size,
                 vocab_index_dict, index_vocab_dict):
        self._text = text
        self._text_size = len(text)
        self._batch_size = batch_size
        self.vocab_size = vocab_size
        self._n_unrollings = n_unrollings
        self.vocab_index_dict = vocab_index_dict
        self.index_vocab_dict = index_vocab_dict

        segment = self._text_size // batch_size

        # number of elements in cursor list is the same as
        # batch_size.  each batch is just the collection of
        # elements in where the cursors are pointing to.
        self._cursor = [ offset * segment for offset in range(batch_size)]
        self._last_batch = self._next_batch()
      
    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size), dtype=np.float)
        for b in range(self._batch_size):
            batch[b] = char2id(self._text[self._cursor[b]], self.vocab_index_dict)
            self._cursor[b] = (self._cursor[b] + 1) % self._text_size
        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
      the last batch of the previous array, followed by num_unrollings new ones.
      """
        batches = [self._last_batch]
        for step in range(self._n_unrollings):
            batches.append(self._next_batch())
        self._last_batch = batches[-1]
        return batches


# Utility functions
def batches2string(batches, index_vocab_dict):
    """Convert a sequence of batches back into their (most likely) string
  representation."""
    s = [''] * batches[0].shape[0]
    for b in batches:
        s = [''.join(x) for x in zip(s, id2char_list(b, index_vocab_dict))]
    return s


def characters(probabilities):
    """Turn a 1-hot encoding or a probability distribution over the possible
    characters back into its (most likely) character representation."""
    return [id2char(c) for c in np.argmax(probabilities, 1)]


def char2id(char, vocab_index_dict):
    try:
        return vocab_index_dict[char]
    except KeyError:
        logging.info('Unexpected char %s', char)
        return 0


def id2char(index, index_vocab_dict):
    return index_vocab_dict[index]

    
def id2char_list(lst, index_vocab_dict):
    return [id2char(i, index_vocab_dict) for i in lst]
