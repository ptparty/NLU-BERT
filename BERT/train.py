from dataset import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from language_model import *
import constant
import sys

class BERTTrainer:
    def __init__(self, transformer, vocab, train_dataloader, valid_dataloader, test_dataloader):
        self.transformer = transformer
        self.model = BERTLM(self.transformer, len(vocab.itos)).to(constant.device)
        self.train_data = train_dataloader
        self.valid_data = valid_dataloader
        self.test_data = test_dataloader
        self.crit = nn.NLLLoss(ignore_index=vocab.pad_index)
        adam = torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.999), weight_decay=constant.weight_decay)
        self.opt = NoamOpt(constant.d_model, 1, constant.warmup_step, adam)
        self.bs = constant.batch_size
        
    def do_train(self, save_path, patience=8, max_epoch=1000):
        best_loss = float('inf')
        count = 0

        for epoch in range(max_epoch):
            self.model.train()
            loss = self._run_epoch(self.train_data, epoch, opt=self.opt)
            print("[%d]Train| loss:%f" %(epoch, loss))
            #self.model.eval()
#             loss = self._run_epoch(self.valid_data, epoch, verbose=True)
#             print("[%d]Valid| loss:%f" %(epoch, loss))

            if loss < best_loss:
                best_loss = loss
                torch.save(self.transformer.state_dict(), save_path)
                count = 0
            else:
                count += 1
            if(count >= patience):
                break
    
    def do_test(self):
        self.model.eval()
        loss = self._run_epoch(self.test_data, verbose=True)
        print("Test| loss:%f" %(loss))

        
    def _run_epoch(self, data_loader, epoch, opt=None, verbose=False):
        "Standard Training and Logging Function"
        total_loss = 0
        data_iter = tqdm(enumerate(data_loader),
                         desc="EP:%d" % (epoch),
                         total=len(data_loader),
                         #bar_format="{l_bar}{r_bar}",
                         file=sys.stdout,
                         disable=verbose
                        )
        
        for i, data in data_iter:
            data = {key: value.to(constant.device) for key, value in data.items()}
            output = self.model(data["bert_input"], data["bert_mask"].unsqueeze(-2))
            loss = self.crit(output.contiguous().view(-1, output.size(-1)), data["bert_label"].contiguous().view(-1))
            
            if opt is not None:
                loss.backward()
                opt.step()
                opt.optimizer.zero_grad()
                
            total_loss += loss.item()
            
#             post_fix = {
#                 "epoch": epoch,
#                 "iter": i,
#                 "avg_loss": total_loss / (i + 1)
#             }
#             data_iter.write(str(post_fix))

        avg_loss = total_loss / len(data_iter)
        return loss