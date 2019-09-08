from sklearn.metrics import f1_score
import torch
import constant
import pickle

def concat_files(filenames, fileiter, out_filename):
    with open(out_filename, 'w') as outfile:
        for i, fname in enumerate(filenames):
            for iteration in range(fileiter[i]):
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)
                        
def write_model_param(save_path, num_vocab, num_entity, num_intent, pad_idx):
    with open(save_path, 'w') as file:
        file.write('num_vocab='+str(num_vocab)+'\n')
        file.write('num_entity='+str(num_entity)+'\n')
        file.write('num_intent='+str(num_intent)+'\n')
        file.write('pad_idx='+str(pad_idx)+'\n')

def load_model_param(load_path):
    with open(load_path, 'r') as file:
        num_vocab = int(file.readline().split('=')[-1])
        num_entity = int(file.readline().split('=')[-1])
        num_intent = int(file.readline().split('=')[-1])
        pad_idx = int(file.readline().split('=')[-1])
    return num_vocab, num_entity, num_intent, pad_idx
        
def loss_compute(x, y, crit):
    loss = crit(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
    return loss

def output_compute(x):
    output = torch.argmax(x, dim=-1)
    return output

def F1_compute_diag(x, y, mask=None, pad_index=None):
    #x,y shape: (nutter, batch, ...)
    #mask shape: (nutter, batch)
    has_3d = True if len(x.size()) == 3 else False
    nu, bs = x.size()[:2]
    x = x.contiguous().tolist()
    y = y.contiguous().tolist()
    x = [x[i][j] for i in range(nu) for j in range(bs) if mask[i][j]]
    y = [y[i][j] for i in range(nu) for j in range(bs) if mask[i][j]]
    if has_3d:
        x = [x[i][j] for i in range(len(x)) for j in range(len(x[i]))]
        y = [y[i][j] for i in range(len(y)) for j in range(len(y[i]))]
    return F1_compute(x, y, pad_index)

def F1_compute(x, y, pad_index=None):
    if pad_index is not None:
        npad_y = [y[idx] for idx in range(len(y)) if y[idx] != pad_index]
        npad_x = [x[idx] for idx in range(len(y)) if y[idx] != pad_index]
        y = npad_y
        x = npad_x
    return f1_score(y, x, average='micro')

def reverse_dict(d):
    new_d = {}
    for key in d.keys():
        new_d[d[key]] = key
    return new_d

def idx_to_orig(idx, d, mask=None):
    batch = idx.size(0)
    origin_list = []
    for i in range(batch):
        if mask is not None:
            l = mask[i].sum().item()
            word_list = []
            for j in range(l):
                word_list.append(d[idx[i][j].item()])
            origin_list.append(word_list)
        else:
            word_list = []
            for j in range(idx[i].size(0)):
                word_list.append(d[idx[i][j].item()])
            origin_list.append(word_list)
    return origin_list      

def print_result_diag(model, dataset, idx, use_turn=True):
    idx_to_word = reverse_dict(dataset.word_to_idx)
    idx_to_slot = reverse_dict(dataset.ent_to_idx)
    idx_to_intent = reverse_dict(dataset.int_to_idx)
    
    data = dataset.__getitem__(idx)
    variables = list(map(lambda var: var.to(constant.device),list(data)))
    variables = list(map(lambda var: var.unsqueeze(0),list(variables)))
    src_seq, src_mask, trg_ent, trg_int = list(map(lambda var: torch.transpose(var, 0, 1), variables))
    is_correct = True
    result = ''
    for u in range(src_seq.size(0)):
        if use_turn:
            pred_ent, pred_int = model.decode(src_seq[u], src_mask[u], turn=u)
        else:
            pred_ent, pred_int = model.decode(src_seq[u], src_mask[u])
        pred_int = output_compute(pred_int)
        s = idx_to_orig(src_seq[u], idx_to_word, src_mask[u])
        mask = src_mask[u].squeeze(1)[:,1:]
        e = idx_to_orig(pred_ent, idx_to_slot, mask)
        g_e = idx_to_orig(trg_ent[u], idx_to_slot, mask)
        i = idx_to_orig(pred_int.unsqueeze(-1), idx_to_intent)
        g_i = idx_to_orig(trg_int[u], idx_to_intent)
        result += "\t Utter #"+str(u)+'\n'
        result += "\t Text: "+str(s)+'\n'
        result += "\t Pred_ent: "+str(e)+'\n'
        result += "\t True_ent: "+str(g_e)+'\n'
        result += "\t Pred_int: "+str(i)+'\n'
        result += "\t True_int: "+str(g_i)+'\n'
        if(i != g_i):
            is_correct = False
    return result, is_correct

def print_wrong_result_diag(model, dataset, use_turn=True):
    idx_to_word = reverse_dict(dataset.word_to_idx)
    idx_to_slot = reverse_dict(dataset.ent_to_idx)
    idx_to_intent = reverse_dict(dataset.int_to_idx)
    results = ''
    for idx in range(len(dataset.inputs)):
        result, is_correct = print_result_diag(model, dataset, idx, use_turn)
        if(not is_correct):
            results += 'Diag #'+str(idx)+'\n'
            results += result
    return results

def save_dict(dictionary,save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

def load_dict(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)