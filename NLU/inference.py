from utils import *
from transformer_MT import *
import constant
import sentencepiece as spm

class InferenceModule:
    def __init__(self, model_path):
        num_vocab, num_entity, num_intent, pad_idx = load_model_param(constant.model_param_save_path)
        self.model = make_MT_model(num_vocab, num_entity, num_intent, pad_idx, N=constant.num_layer, n_utter=constant.utter_window).to(constant.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load('{}.model'.format(constant.prefix))
        self.token_to_idx = load_dict('word_to_idx')
        self.idx_to_token = reverse_dict(self.token_to_idx)
        self.idx_to_entity = reverse_dict(load_dict('entity_to_idx'))
        self.idx_to_intent = reverse_dict(load_dict('intent_to_idx'))
        
        self.turn = 0
        
    def inference(self, raw_seq):
        src = self._preprocess(raw_seq).to(constant.device)
        src = src.unsqueeze(0)
        pred_ent, pred_int = self.model.decode(src, turn=self.turn)
        pred_int = output_compute(pred_int)

        i = idx_to_orig(pred_int.unsqueeze(-1), self.idx_to_intent)
        e = idx_to_orig(pred_ent, self.idx_to_entity)
        s = idx_to_orig(src, self.idx_to_token)
        result = dict()
        result['text'] = s[0][1:]
        result['intent'] = i[0]
        result['entity'] = e[0]
        self.turn += 1
        return result
    
    def reset_turn(self, turn=0):
        self.turn = turn
        
    def _preprocess(self, raw_seq):
        tokens = self.tokenizer.EncodeAsPieces(raw_seq)
        idxs = []
        for token in tokens:
            if token in self.token_to_idx:
                idxs.append(self.token_to_idx[token])
            else:
                idxs.append(self.token_to_idx['[UNK]'])
        idxs.insert(0, self.token_to_idx['[CLS]'])
        idxs = torch.tensor(idxs, dtype=torch.long)
        return idxs
