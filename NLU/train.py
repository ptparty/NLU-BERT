from data_process import *
from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
import sys

class TrainModule:
    def __init__(self, model, data, tokenizer, crit, opt):
        self.model = model
        self.crit = crit
        self.opt = opt
        self.bs = constant.batch_size

        self.train_data =  Dynamic_Handler(data, 0, tokenizer)
        self.valid_data = Dynamic_Handler(data, 1, tokenizer)
        self.test_data = Dynamic_Handler(data, 2, tokenizer)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=self.bs, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        self.valid_loader = DataLoader(dataset=self.valid_data, batch_size=self.bs, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=self.bs, shuffle=True, collate_fn=collate_fn, pin_memory=True)
        
        
    def do_train(self, save_path, int_coef=1.0, ent_coef=1.0, patience=8, max_epoch=1000):
        best_loss = float('inf')
        count = 0

        for epoch in range(max_epoch):
            self.model.train()
            loss_ent, loss_int, f1_ent, f1_int = \
                        self._run_epoch(self.train_loader, opt=self.opt, int_coef=int_coef, ent_coef=ent_coef)
            print("[%d]Train| E_loss:%f | I_loss:%f | E_F1:%f | I_F1:%f" %(epoch, loss_ent, loss_int, f1_ent, f1_int))
            self.model.eval()
            loss_ent, loss_int, f1_ent, f1_int = \
                        self._run_epoch(self.valid_loader, int_coef=int_coef, ent_coef=ent_coef, verbose=True)
            print("[%d]Valid| E_loss:%f | I_loss:%f | E_F1:%f | I_F1:%f" %(epoch, loss_ent, loss_int, f1_ent, f1_int))

            loss = loss_ent + loss_int
            if loss < best_loss:
                best_loss = loss
#                 torch.save(self.model.state_dict(), save_path)
                count = 0
            else:
                count += 1
            if(count >= patience):
                break
    
    def do_test(self, int_coef=1.0, ent_coef=1.0):
        self.model.eval()
        loss_ent, loss_int, f1_ent, f1_int = \
                        self._run_epoch(self.test_loader, int_coef=int_coef, ent_coef=ent_coef, verbose=True)
        print("Test| E_loss:%f | I_loss:%f | E_F1:%f | I_F1:%f" %(loss_ent, loss_int, f1_ent, f1_int))

        
    def _run_epoch(self, data_loader, opt=None, int_coef=1.0, ent_coef=1.0, verbose=False):
        "Standard Training and Logging Function"
        total_norm = 0
        total_loss_ent = 0
        total_loss_int = 0
        sum_f1_ent = 0
        sum_f1_int = 0
        pad_idx = 0

        with tqdm(total=len(data_loader), file=sys.stdout, mininterval=5, disable=verbose) as pbar:
            for i, data in enumerate(data_loader):
                #src_seq shape: (batch_size, max_utter, max_word) -> (max_utter, batch_size, max_word)
                #utt_mask shape: (batch_size, max_utter) -> (max_utter, batch_size)
                variables = list(map(lambda var: var.to(constant.device),list(data)))
                src_seq, src_mask, trg_ent, trg_int, utt_mask = list(map(lambda var: torch.transpose(var, 0, 1), variables))
                nutters, nbatchs = src_seq.size()[:2]
                pred_ent_list = FloatTensor().type_as(trg_ent)
                pred_int_list = FloatTensor().type_as(trg_int)
                for i in range(nutters):
                    #ntokens, loss_ent, loss_int shape: (batch_size)
                    #pred_int shape: (batch_size)
                    #pred_ent shape: (batch_size, max_word)
                    pred_ent, utt_loss_ent, pred_int = self.model(src_seq[i], trg_ent[i], src_mask[i], turn=i)
                    utt_loss_int = loss_compute(pred_int, trg_int[i], self.crit)
                    pred_int = output_compute(pred_int)
                    pred_ent_list = torch.cat([pred_ent_list,pred_ent.unsqueeze(0)], dim=0)
                    pred_int_list = torch.cat([pred_int_list,pred_int.unsqueeze(0)], dim=0)

                    is_utter = utt_mask[i]
                    norm = is_utter.sum()
                    utt_loss_ent = (utt_loss_ent * is_utter).sum()
                    utt_loss_int = (utt_loss_int * is_utter).sum()
                    loss = (ent_coef * utt_loss_ent / norm) + (int_coef * utt_loss_int / norm)
        
                    if opt is not None:
                        loss.backward()
                        opt.step()
                        opt.optimizer.zero_grad()

                    total_loss_ent += utt_loss_ent.item()
                    total_loss_int += utt_loss_int.item()
                    total_norm += norm.item()

                f1_ent = F1_compute_diag(pred_ent_list, trg_ent, mask=utt_mask, pad_index=pad_idx)
                f1_int = F1_compute_diag(pred_int_list, trg_int, mask=utt_mask)
                sum_f1_ent += f1_ent
                sum_f1_int += f1_int
                pbar.update()

            avg_loss_ent = total_loss_ent*ent_coef / total_norm
            avg_loss_int = total_loss_int*int_coef / total_norm
            avg_f1_ent = sum_f1_ent / len(data_loader)
            avg_f1_int = sum_f1_int / len(data_loader)

        return avg_loss_ent, avg_loss_int, avg_f1_ent, avg_f1_int