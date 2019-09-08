from transformer_MT import *
import constant

class TFNLU(nn.Module):
    """
    A standard Encoder architecture. Base for this and many
    other models.
    """
    def __init__(self, transformer, trg_intent, trg_entity, pad_idx, cuda=True, load_weight=True):
        super(TFNLU, self).__init__()
        self.transformer = transformer
        self.int_cls = IntentPrediction(constant.d_model, trg_intent)
        self.ent_cls = EntityPrediction(constant.d_model, trg_entity)
        self.crf = CRF(trg_entity, pad_idx, batch_first=True, reduction='none', cuda=cuda)
        
        self.init_param(self.int_cls)
        self.init_param(self.ent_cls)
        self.init_param(self.crf)
        if load_weight:
            load_path = os.path.join(constant.BERT_model_path, 'bert.pth')
            self.transformer.load_state_dict(torch.load(load_path))
        else:
            self.init_param(self.transformer)
            
    def init_param(self, model):
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, trg, src_mask=None, turn=0):
        "Take in and process masked src and target sequences."
        output = self.transformer(src, src_mask, turn)
        pred_int = F.log_softmax(self.int_cls(output), dim=-1)
        output_ent = self.ent_cls(output)
        if src_mask is not None:
            src_mask = src_mask.squeeze(1)[:,1:]
        llh_ent = -self.crf(output_ent, trg, src_mask) 
        pred_ent = self.crf.decode(output_ent, src_mask)
        return pred_ent, llh_ent, pred_int
    
    def decode(self, src, src_mask=None, turn=0):
        output = self.encode(src, src_mask, turn)
        pred_int = F.log_softmax(self.int_cls(output), dim=-1)
        output_ent = self.ent_cls(output)
        if src_mask is not None:
            src_mask = src_mask.squeeze(1)[:,1:]
        pred_ent = self.crf.decode(output_ent, src_mask)
        return pred_ent, pred_int
    
    
class IntentPrediction(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(IntentPrediction, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x[:,0])
        
class EntityPrediction(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(EntityPrediction, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x[:,1:])
        
'''
def make_MT_model(src_vocab, trg_entity, trg_intent, pad_idx, cuda=True, load_weight=False):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    N = constant.N
    n_utter = constant.n_utter
    d_model = constant.d_model
    d_ff = constant.d_ff
    h = constant.h
    dropout= constant.dropout
    
    position = PositionalEncoding(d_model, dropout)
    attn = MultiHeadedAttention(h, d_model)
    mt_attn = MultiTurnAttention(c(position), h, d_model, n_utter)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    tf = MT_Transformer(
        MT_Encoder(MT_EncoderLayer(d_model, attn, mt_attn, ff, dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position))
    )
    model = MT_TransformerCrf(
        tf,
        CRF(trg_entity, pad_idx, batch_first=True, reduction='none', cuda=cuda),
        IntentPrediction(d_model, trg_intent),
        EntityPrediction(d_model, trg_entity)
    )
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
        
    if load_weight:
        save_path = os.path.join(constant.BERT_model_path, 'bert.pth')
        tf.load_state_dict(torch.load(save_path))
    return model
'''