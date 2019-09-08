import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_SP = False
SP_prefix = 'SP'
SP_path = './SP_data/sp.txt'
vocab_size = 15000

NLU_data_path = './NLU_data'
NLU_model_path = './NLU/saved_model' 
NLU_param_path = './NLU/saved_model/model_param.txt'

BERT_data_path = './BERT_data'
BERT_model_path = './BERT/saved_model' 
BERT_param_path = './BERT/saved_model/model_param.txt'
BERT_mask_prob = 0.2

seq_len = 50
batch_size = 768

N = 3
n_utter = 5 
d_model = 512
d_ff = 2048
h = 8
dropout = 0.1

warmup_step=10000
weight_decay=0.01