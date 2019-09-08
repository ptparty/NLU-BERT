from torch.utils.data import Dataset
from torch import FloatTensor
import os
import torch
import torch.nn as nn
import constant
from utils import *

class Multi_Turn_Processor():
    def __init__(self, data_dir, word_to_idx):
        self.entities = self.get_entity_labels_from_files(data_dir)
        self.intents = self.get_intent_labels_from_files(data_dir)
        self.entity_to_idx = self.get_label_to_id(self.entities)
        self.intent_to_idx = self.get_label_to_id(self.intents)
        self.word_to_idx = word_to_idx
        
        self.train = self.get_train_examples(data_dir)
        self.valid = self.get_dev_examples(data_dir)
        self.test = self.get_test_examples(data_dir) 
        self.num_train = len(self.train)
        self.num_valid = len(self.valid)
        self.num_test = len(self.test)
        self.data = [self.train, self.valid, self.test] 
        
        self.max_utter, self.max_word = max(self.get_max_len(self.train),self.get_max_len(self.valid),self.get_max_len(self.test))     
        self.save_params()
    
    def get_params(self):
        return (len(self.word_to_idx), len(self.entity_to_idx), len(self.intent_to_idx), self.word_to_idx['[PAD]'])
        
    def save_params(self):
        write_model_param(constant.NLU_param_path, len(self.word_to_idx), len(self.entity_to_idx), len(self.intent_to_idx), self.word_to_idx['[PAD]'])
        save_dict(self.word_to_idx, os.path.join(constant.NLU_model_path, 'word_to_idx.pkl'))
        save_dict(self.entity_to_idx, os.path.join(constant.NLU_model_path, 'entity_to_idx.pkl'))
        save_dict(self.intent_to_idx, os.path.join(constant.NLU_model_path, 'intent_to_idx.pkl'))
    
    def get_examples(self, data_dir):
        path_seq_in = os.path.join(data_dir, "seq.in")
        path_seq_out = os.path.join(data_dir, "seq.out")
        path_label = os.path.join(data_dir, "label")
        seq_in_list, seq_out_list, label_list = [], [], []
        i = -1
        
        with open(path_seq_in) as seq_in_f:
            with open(path_seq_out) as seq_out_f:
                with open(path_label) as label_f:
                    for seqin, seqout, label in zip(seq_in_f.readlines(), seq_out_f.readlines(), label_f.readlines()):
                        seqin_words = [word for word in seqin.split() if len(word) > 0]
                        seqout_words = [word for word in seqout.split() if len(word) > 0]
                        assert len(seqin_words) == len(seqout_words)
                        
                        if seqin_words[0] == '#dialog':
                            seq_in_list.append([])
                            seq_out_list.append([])
                            label_list.append([])
                            i += 1
                            continue
                        
                        CE = self.check_entity(seqout_words)
                        CI = self.check_intent(label)
                        if CE and CI:
                            seq_in_list[i].append(seqin_words)
                            seq_out_list[i].append(seqout_words)
                            label_list[i].append(label.strip().replace("\n", ""))
            lines = list(zip(seq_in_list, seq_out_list, label_list))
            return lines
    
    def check_entity(self, seq):
        for e in seq:
            if e not in self.entities:
                return False
        return True
    
    def check_intent(self, intent):
        if intent.strip() not in self.intents:
            return False
        else:
            return True
    
    def get_train_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "train"))

    def get_dev_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "valid"))

    def get_test_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "test"))

    def get_entity_labels_from_files(self, data_dir):
        label_set = set()
        f_type = "train"
        seq_out_dir = os.path.join(os.path.join(data_dir, f_type), "seq.out")
        with open(seq_out_dir) as data_f:
            seq_sentence_list = [seq.split() for seq in data_f.readlines() if seq.find('#dialog') < 0]
            seq_word_list = [word for seq in seq_sentence_list for word in seq]
            label_set = set(seq_word_list)
        
        I_label_set = set()
        for label in label_set:
            s_label = label.split('-')
            if((len(s_label) > 1) and (s_label[0] == 'B')):
                entity_type = s_label[1]
                I_entity_type = 'I-' + entity_type
                I_label_set.add(I_entity_type)
                
        label_set = label_set | I_label_set
        label_list = list(label_set)
        label_list.sort()
        return ['[PAD]'] + label_list

    def get_intent_labels_from_files(self, data_dir, cut_count=15):
        intents = list()
        f_type = "train"
        seq_out_dir = os.path.join(os.path.join(data_dir, f_type), "label")
        with open(seq_out_dir) as data_f:
            intent_list = [intent.strip() for intent in data_f.readlines() if intent.find('#dialog') < 0]
            intent_set = set(intent_list)
            intents = [intent for intent in intent_set if intent_list.count(intent) > cut_count]
            intents.sort()
        return intents
    
    def get_max_len(self, dataset):
        dataset = list(zip(*dataset))
        dialogs = dataset[0]
        max_utter = 0
        max_word = 0
        
        for dialog in dialogs:
            num_utter = len(dialog)
            max_utter = num_utter if max_utter < num_utter else max_utter
            for utter in dialog:
                num_word = len(utter)
                max_word = num_word if max_word < num_word else max_word
        return max_utter, max_word

    def get_label_to_id(self, labels):
        label_to_idx = {}
        for label in labels:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
        return label_to_idx
    
        
class Dynamic_Handler(Dataset):
    def __init__(self, data_handler, data_type, tokenizer):
        data = data_handler.data[data_type]
        data = list(zip(*data))
        self.inputs = data[0]
        self.entities = data[1]
        self.intents = data[2]
        self.max_word = data_handler.max_word * 2
        self.word_to_idx = data_handler.word_to_idx
        self.ent_to_idx = data_handler.entity_to_idx 
        self.int_to_idx = data_handler.intent_to_idx
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        #text, entity: 2d list (num_utter, num_word)
        text = self.inputs[index]
        entity = self.entities[index]
        intent = self.intents[index]
        num_utter = len(text)
        src_seq_list = []
        src_mask_list = []
        trg_ent_list = []
        trg_int_list = []

        for i in range(num_utter):
            raw_seq = ' '.join(text[i])
            src_seq = self.tokenizer.EncodeAsPieces(raw_seq)
            trg_ent = self.extend_trg(src_seq, entity[i])
            src_seq = self.preprocess(src_seq, self.word_to_idx)
            src_mask = (src_seq != self.word_to_idx['[PAD]'])
            trg_ent = self.preprocess(trg_ent, self.ent_to_idx, trg=True)
            trg_int = torch.tensor([self.int_to_idx[intent[i]]],dtype=torch.long)
            src_seq_list.append(src_seq)
            src_mask_list.append(src_mask)
            trg_ent_list.append(trg_ent)
            trg_int_list.append(trg_int)
        
        src_seqs = torch.stack(src_seq_list)
        src_masks = torch.stack(src_mask_list)
        trg_ents = torch.stack(trg_ent_list)
        trg_ints = torch.stack(trg_int_list)
        return src_seqs, src_masks, trg_ents, trg_ints
    
    def extend_trg(self, src_seq, trg_ent):
        tmp_ent = list(trg_ent)
        for i, w in enumerate(src_seq):
            if w[0] != chr(9601):
                prev_trg = tmp_ent[i-1]
                if(prev_trg == 'O'):
                    curr_trg = 'O'
                else:
                    BIO_tag, entity_type = prev_trg.split('-')
                    curr_trg = 'I-'+ entity_type
                tmp_ent.insert(i, curr_trg)
        return tmp_ent

    def preprocess(self, tokens, token_dict, trg=False):
        max_word = self.max_word
        idxs = []
        for token in tokens:
            if token in token_dict:
                idxs.append(token_dict[token])
            else:
                idxs.append(token_dict['[UNK]'])
                
        if not trg:
            idxs.insert(0, token_dict['[CLS]'])
        else:
            max_word -= 1
            
        if(len(idxs) >= max_word):
            idxs = idxs[:max_word]
        else:
            num_pad = max_word - len(idxs)
            idxs.extend([token_dict['[PAD]'] for _ in range(num_pad)])
        idxs = torch.tensor(idxs, dtype=torch.long)
        return idxs
    
    
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (src_seq_list, src_mask_list, trg_ent_list, trg_int_list). 
            - src_seq: torch tensor of shape [nutter, max_word],
            - src_mask: torch tensor of shape [nutter, max_word],
            - trg_ent: torch tensor of shape [nutter, max_word-1],
            - trg_int: torch tensor of shape [nutter, 1].
    Returns:
    """
    def make_utter_mask(num_utt):
        index = torch.arange(num_utt)
        mask = torch.zeros(max_utt).index_fill_(0, index, 1.0)
        return mask
    
    src_seqs, src_masks, trg_ents, trg_ints = list(map(list, zip(*data)))
    max_utt = max([src_seq.size(0) for src_seq in src_seqs])
    bs = len(src_seqs)
    utt_pad = 1.0
    utt_masks = []
    for i in range(bs):
        num_utt = src_seqs[i].size(0)
        padding_utt = nn.ConstantPad2d((0,0,0,max_utt-num_utt), utt_pad)
        src_seqs[i] = padding_utt(src_seqs[i])
        src_masks[i] = padding_utt(src_masks[i])
        trg_ents[i] = padding_utt(trg_ents[i])
        trg_ints[i] = padding_utt(trg_ints[i])
        utt_masks.append(make_utter_mask(num_utt))
        
    src_seqs = torch.stack(src_seqs)
    src_masks = torch.stack(src_masks).unsqueeze(-2)
    trg_ents = torch.stack(trg_ents)
    trg_ints = torch.stack(trg_ints).squeeze()
    utt_masks = torch.stack(utt_masks)
    
    return src_seqs, src_masks, trg_ents, trg_ints, utt_masks

'''
class Multi_Turn_Processor():
    def __init__(self, data_dir, vocabs):
        self.train = self.get_train_examples(data_dir)
        self.valid = self.get_dev_examples(data_dir)
        self.test = self.get_test_examples(data_dir) 
        self.num_train = len(self.train)
        self.num_valid = len(self.valid)
        self.num_test = len(self.test)
        self.data = [self.train, self.valid, self.test] 
        
        self.entities = self.get_entity_labels_from_files(data_dir)
        self.intents = self.get_intent_labels_from_files(data_dir)
        
        self.max_utter, self.max_word = max(self.get_max_len(self.train),self.get_max_len(self.valid),self.get_max_len(self.test))
        self.entity_to_idx = self.get_label_to_id(self.entities)
        self.intent_to_idx = self.get_label_to_id(self.intents)
        self.word_to_idx = self.get_word_to_id(vocabs)     
        self.save_params()
    
    def get_params(self):
        return (len(self.word_to_idx), len(self.entity_to_idx), len(self.intent_to_idx), self.word_to_idx['[PAD]'])
        
    def save_params(self):
        write_model_param(constant.model_param_save_path, len(self.word_to_idx), len(self.entity_to_idx), len(self.intent_to_idx), self.word_to_idx['[PAD]'])
        save_dict(self.word_to_idx, 'word_to_idx')
        save_dict(self.entity_to_idx, 'entity_to_idx')
        save_dict(self.intent_to_idx, 'intent_to_idx')
    
    def get_examples(self, data_dir):
        path_seq_in = os.path.join(data_dir, "seq.in")
        path_seq_out = os.path.join(data_dir, "seq.out")
        path_label = os.path.join(data_dir, "label")
        seq_in_list, seq_out_list, label_list = [], [], []
        i = -1
        
        with open(path_seq_in) as seq_in_f:
            with open(path_seq_out) as seq_out_f:
                with open(path_label) as label_f:
                    for seqin, seqout, label in zip(seq_in_f.readlines(), seq_out_f.readlines(), label_f.readlines()):
                        seqin_words = [word for word in seqin.split() if len(word) > 0]
                        seqout_words = [word for word in seqout.split() if len(word) > 0]
                        assert len(seqin_words) == len(seqout_words)
                        
                        if seqin_words[0] == '#dialog':
                            seq_in_list.append([])
                            seq_out_list.append([])
                            label_list.append([])
                            i += 1
                            continue
                        seq_in_list[i].append(seqin_words)
                        seq_out_list[i].append(seqout_words)
                        label_list[i].append(label.replace("\n", ""))
            lines = list(zip(seq_in_list, seq_out_list, label_list))
            return lines

    def get_train_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "train"))

    def get_dev_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "valid"))

    def get_test_examples(self, data_dir):
        return self.get_examples(os.path.join(data_dir, "test"))

    def get_entity_labels_from_files(self, data_dir):
        label_set = set()
        for f_type in ["train", "valid", "test"]:
            seq_out_dir = os.path.join(os.path.join(data_dir, f_type), "seq.out")
            with open(seq_out_dir) as data_f:
                seq_sentence_list = [seq.split() for seq in data_f.readlines() if seq.find('#dialog') < 0]
                seq_word_list = [word for seq in seq_sentence_list for word in seq]
                label_set = label_set | set(seq_word_list)
        
        I_label_set = set()
        for label in label_set:
            s_label = label.split('-')
            if((len(s_label) > 1) and (s_label[0] == 'B')):
                entity_type = s_label[1]
                I_entity_type = 'I-' + entity_type
                I_label_set.add(I_entity_type)
                
        label_set = label_set | I_label_set
        label_list = list(label_set)
        label_list.sort()
        return ["[PAD]"] + label_list
    
    def get_intent_labels_from_files(self, data_dir):
        label_set = set()
        for f_type in ["train", "valid", "test"]:
            seq_out_dir = os.path.join(os.path.join(data_dir, f_type), "label")
            with open(seq_out_dir) as data_f:
                intent_list = [intent.strip() for intent in data_f.readlines() if intent.find('#dialog') < 0]
                label_set = label_set | set(intent_list)
        label_list = list(label_set)
        label_list.sort()
        return label_list
    
    def get_max_len(self, dataset):
        dataset = list(zip(*dataset))
        dialogs = dataset[0]
        max_utter = 0
        max_word = 0
        
        for dialog in dialogs:
            num_utter = len(dialog)
            max_utter = num_utter if max_utter < num_utter else max_utter
            for utter in dialog:
                num_word = len(utter)
                max_word = num_word if max_word < num_word else max_word
        return max_utter, max_word
    
    def get_word_to_id(self, vocabs):
        word_to_idx = {}
        word_to_idx['[PAD]'] = 0
        word_to_idx['[CLS]'] = 1
        word_to_idx['[UNK]'] = 2
        for word in vocabs:
            word_to_idx[word] = len(word_to_idx)
        return word_to_idx
    
    def get_label_to_id(self, labels):
        label_to_idx = {}
        for label in labels:
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
        return label_to_idx
        '''
    