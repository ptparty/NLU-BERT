from transformer import *

class MT_Transformer(nn.Module):
    def __init__(self, src_vocab):
        super(MT_Transformer, self).__init__()
        
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
        
        self.encoder = MT_Encoder(MT_EncoderLayer(d_model, attn, mt_attn, ff, dropout), N)
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        
    def forward(self, src, mask, turn):
        return self.encoder(self.src_embed(src), mask, turn)
    
class MT_Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(MT_Encoder, self).__init__()

        #clone N different weight layers
        #and one unique norm layer
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, turn):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, turn)
        return self.norm(x)

class MT_EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, turn_attn, feed_forward, dropout):
        super(MT_EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.turn_attn = turn_attn
        self.feed_forward = feed_forward

        #Why deepcopy the SublayerConnection layers?
        #To make different batch nomalization parameters for each sublayer
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, mask, turn):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        if turn > -1:
            x = self.sublayer[1](x, lambda x: self.turn_attn(x, turn))
        return self.sublayer[2](x, self.feed_forward)
    
class MultiTurnAttention(nn.Module):
    def __init__(self, pe, h, d_model, n_utter, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiTurnAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.pe = pe
        self.d_k = d_model // h
        self.n_u = n_utter
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.queue = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        #[in]query shape: (batch, h, 1, d_k)
        #[in]key,value shape: (batch, h, n_u, d_k)
        #[in]mask shape: (1, 1, 1, n_u) 
        #[var]score shape: (batch, h, 1, n_u)
        #[out]return shape: (batch, h, 1, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def turn_mask(self, turn):
        "make turn_mask"
        #[var]attn_shape: (1, 1, 1, n_u) -> For broadcasting mask to (batch, h, 1, n_u)
        attn_shape = (1, 1, 1, self.n_u)
        index = torch.arange(turn + 1)
        t_mask = torch.zeros(attn_shape, dtype=torch.uint8).index_fill_(-1, index, 1).to(constant.device)
        return t_mask
    
    def insert_queue(self, query):
        "push current query to queue and pop the oldest query"
        assert self.queue is not None
        self.queue = torch.cat((query.detach(), self.queue[:,:-1]), dim = 1)

    def forward(self, H, turn):
        #H shape: (batch, time_step, dimension)
        #query shape: (batch, 1, dimension)
        #queue shape: (batch, max_utter, dimension)
        query = H[:,0].unsqueeze(1)
        nbatches, nwords, dmodel = H.size()
        mask = None
        
        if turn == 0:
            self.queue = Variable(torch.zeros([nbatches, self.n_u, dmodel]).type_as(query)
                                  ,requires_grad=False)
        self.insert_queue(query)
        
        #[In case]zero paddings exist in queue
        if turn < self.n_u - 1 :
            mask = self.turn_mask(turn)   
        
        #Do all the linear projections in batch from d_model => h x d_k
        Q_PE = self.pe(self.queue)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, Q_PE, Q_PE))]

        #Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        #"Concat" using a view and apply a final linear.
        #x shape: (batch, 1, dimension)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        x = self.linears[-1](x)
        pad_func = nn.ConstantPad2d((0,0,0,nwords-1), 0.0)
        x = pad_func(x)
        return x
    
class MultiTurnAttention_v2(nn.Module):
    def __init__(self, pe, h, d_model, n_utter, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiTurnAttention_v2, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.pe = pe
        self.d_k = d_model // h
        self.n_u = n_utter
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.queue = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        #[in]query shape: (batch, h, 1, d_k)
        #[in]key,value shape: (batch, h, n_u, d_k)
        #[in]mask shape: (1, 1, 1, n_u) 
        #[var]score shape: (batch, h, 1, n_u)
        #[out]return shape: (batch, h, 1, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn
    
    def turn_mask(self, turn):
        "make turn_mask"
        #[var]attn_shape: (1, 1, 1, n_u) -> For broadcasting mask to (batch, h, 1, n_u)
        attn_shape = (1, 1, 1, self.n_u)
        index = torch.arange(turn + 1)
        t_mask = torch.zeros(attn_shape, dtype=torch.uint8).index_fill_(-1, index, 1).to(constant.device)
        return t_mask
    
    def insert_queue(self, query):
        "push current query to queue and pop the oldest query"
        assert self.queue is not None
        self.queue = torch.cat((query.detach(), self.queue[:,:-1]), dim = 1)

    def forward(self, H, turn):
        #H shape: (batch, time_step, dimension)
        #query shape: (batch, 1, dimension)
        #queue shape: (batch, max_utter, dimension)
        query = H[:,0].unsqueeze(1)
        nbatches, nwords, dmodel = H.size()
        mask = None
        
        if turn == 0:
            self.queue = Variable(torch.zeros([nbatches, self.n_u, dmodel]).type_as(query)
                                  ,requires_grad=False)
        self.insert_queue(query)
        
        #[In case]zero paddings exist in queue
        if turn < self.n_u - 1 :
            mask = self.turn_mask(turn)   
        
        #Do all the linear projections in batch from d_model => h x d_k
        Q_PE = self.pe(self.queue)
        q, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, Q_PE, Q_PE))]

        #Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(q, key, value, mask=mask, dropout=self.dropout)

        #"Concat" using a view and apply a final linear.
        #x shape: (batch, 1, dimension)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        gate = F.sigmoid(self.linears[-1](x - query))
        x *= gate
        pad_func = nn.ConstantPad2d((0,0,0,nwords-1), 0.0)
        x = pad_func(x)
        return x