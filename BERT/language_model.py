from model.transformer_MT import *
        
class BERTLM(nn.Module):
    """
    A standard Encoder architecture. Base for this and many
    other models.
    """
    def __init__(self, transformer, trg_vocab_size):
        super(BERTLM, self).__init__()
        self.transformer = transformer
        self.mask_cls = MaskedLanguageModel(constant.d_model, trg_vocab_size)
        self.vocab_size = trg_vocab_size
        self.init_param()
        
    def forward(self, src, mask=None, turn=-1):
        "Take in and process masked src and target sequences."
        output = self.transformer(src, mask, turn)
        pred = F.log_softmax(self.mask_cls(output), dim=-1)
        return pred
    
    def init_param(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super(MaskedLanguageModel, self).__init__()
        self.proj = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        return self.proj(x)