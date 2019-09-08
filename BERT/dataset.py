from torch.utils.data import Dataset
import tqdm
import torch
import random
import constant

class BERTHandler(Dataset):
    def __init__(self, corpus_path, vocab, tokenizer, seq_len, encoding="utf-8"):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            self.lines = f.readlines()
            self.corpus_lines = len(self.lines)    

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        line = self.lines[item]
        random, label = self.random_word(line)

        random = [self.vocab.cls_index] + random
        label = [self.vocab.pad_index] + label
        
        bert_input = random[:self.seq_len]
        bert_label = label[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding)
        bert_label.extend(padding)
        bert_mask = [1 if e != self.vocab.pad_index else 0 for e in bert_input]

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "bert_mask" : bert_mask
                 }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = self.tokenizer.EncodeAsPieces(sentence)
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < constant.BERT_mask_prob :
                prob /= constant.BERT_mask_prob 

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(len(self.vocab))

                # 10% randomly change token to current token
                else:
                    tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

                output_label.append(self.vocab.stoi.get(token, self.vocab.unk_index))

            else:
                tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
                output_label.append(self.vocab.pad_index)

        return tokens, output_label