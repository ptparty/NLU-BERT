import pickle
import tqdm
import sys
from collections import Counter



class TorchVocab(object):
    def __init__(self, vocab, specials=['<pad>', '<oov>']):
        self.itos = list(specials)
        for word in vocab:
            self.itos.append(word)
        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class WordVocab(TorchVocab):
    def __init__(self, vocab):
        self.pad_index = 0
        self.cls_index = 1
        self.unk_index = 2
        self.mask_index = 3
        super().__init__(vocab, specials=["[PAD]", "[CLS]", "[UNK]", "[MASK]"])

    def to_seq(self, sentence, tokenizer, seq_len=None, with_cls=False, with_len=False):
        if isinstance(sentence, str):
            sentence = tokenizer.EncodeAsPieces(sentence)

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_cls:
            seq = [self.cls_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)
