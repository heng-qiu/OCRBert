import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from .OCRB import OCRBLM,OCRB
from tqdm import tqdm

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def mask_tokens(inputs, vocab, mask_id, prob=0.15):
    """ 准备掩码语言模型的掩码输入和标签：
        80% 为 MASK,10% 为随机,10% 保留原值。
        只对非零值进行掩码。
    """
    labels = inputs.clone()

    # 只对非零值进行掩码
    non_zero_indices = inputs != 0  # 判断哪些位置的值不为零
    masked_indices = torch.bernoulli(torch.full(labels.shape, prob)).bool() & non_zero_indices

    # 对于没有被掩码的位置，标签设为 -1（不做掩码）
    labels[~masked_indices] = -1  
    
    # 80% 的掩码位置，用 MASK 代替
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_id

    # 10% 的掩码位置，替换为随机词
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(vocab), labels.shape, dtype=torch.long)  # 确保随机词在词汇表范围内
    inputs[indices_random] = random_words[indices_random]

    # 剩下的 10% 保持原始词不变，由 `labels` 处理
    return inputs, labels

class OCRBDataset(Dataset):
    def __init__(self, data, vocab=None, mask_id=None, prob=None):
        self.data = data
        self.vocab = vocab
        self.mask_id = mask_id
        self.prob=prob
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        if self.vocab is not None and self.mask_id is not None:
            inputs,label = mask_tokens(torch.from_numpy(self.data[item].toarray().astype(int)), self.vocab, self.mask_id)
        else:
            inputs=torch.from_numpy(self.data[item].toarray().astype(int))
            label=inputs.clone()
        output = {"ocrb_input": inputs,
                  "ocrb_label": label}
        # return {key: torch.tensor(value) for key, value in output.items()}
        return output

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
            
class OCRBTrainer:
    def __init__(self, vocab_size: int, positions,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 hidden=24, 
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.0001, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 20):
        """
        :param OCRB: OCRB model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: training with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        # Setup cuda device for OCRB training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        if cuda_devices:
            self.device = torch.device(cuda_devices)
        else:
            self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.hidden = hidden
        self.positions = torch.from_numpy(positions).to(self.device)
        # This OCRB model will be saved every epoch
        self.OCRB = OCRB(vocab_size,hidden=hidden)
        # Initialize the OCRB Language Model, with OCRB model
        self.model = OCRBLM(self.OCRB, vocab_size).to(self.device)

        # # Distributed GPU training if CUDA can detect more than 1 GPU
        # if with_cuda and torch.cuda.device_count() > 1:
        #     print("Using %d GPUS for OCRB" % torch.cuda.device_count())
        #     self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.OCRB.hidden, n_warmup_steps=warmup_steps)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=-1)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            # if len(data['ocrb_label'].shape)==3:
            #     data['ocrb_input']=torch.squeeze(data['ocrb_input'], dim=1)
            #     data['ocrb_label']=torch.squeeze(data['ocrb_label'], dim=1)
            # data = {key: value.to(self.device) for key, value in data.items()}
            # 1. forward the next_sentence_prediction and masked_lm model
            inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)
            
            # mask_lm_output = self.model.forward(data["ocrb_input"])
            # 2-2. NLLLoss of predicting masked token word
            # mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["ocrb_label"])
            label=torch.squeeze(data['ocrb_label'], dim=1).to(self.device).detach()
            
            mask_lm_output, _ = self.model.forward(inputs,self.positions)
            loss = self.criterion(mask_lm_output.transpose(1, 2), label)

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # next sentence prediction accuracy
            # correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            # total_correct += correct
            # total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                # "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))
        # print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
            #   total_correct * 100.0 / total_element)
        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter))

    def save(self, epoch, file_path="output/OCRB_trained.model"):
        """
        Saving the current OCRB model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.OCRB.cpu(), output_path)
        self.OCRB.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path
    
    @torch.no_grad()
    def get_results(self, data_loader: DataLoader,positions):
        outputs = [] 
        for data in tqdm(data_loader):
            # inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
            inputs=torch.squeeze(data['ocrb_input'], dim=1)
            inputs=inputs.to('cuda:1')
            output, _ = self.model.ocrb.forward(inputs,positions)
            outputs.append(output.cpu())
        return torch.cat(outputs, dim=0)
    
    @torch.no_grad()
    def get_all_embed(self, data_loader: DataLoader,positions,gene_pos):
        output_gene = []
        output_mean = [] 
        output_w = []
        output_bw = []
        for data in tqdm(data_loader):
            inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
            # inputs=torch.squeeze(data['ocrb_input'], dim=1)
            # inputs=inputs.to('cuda:1')
            output, _ = self.model.ocrb.forward(inputs,positions)
            output_gene.append(torch.squeeze(output[:,gene_pos,:], dim=1))
            # binary_tensor = (inputs[:, gene_pos] != 0).to(torch.float32)
            output_mean.append(torch.mean(output, dim=1))
            binary_tensor = (inputs!= 0).to(torch.float32)
            output_w.append(torch.mean(output*torch.unsqueeze(inputs, dim=-1),dim=1))
            output_bw.append(torch.mean(output*torch.unsqueeze(binary_tensor, dim=-1),dim=1))
            
        return torch.cat(output_gene, dim=0), torch.cat(output_mean, dim=0), torch.cat(output_w, dim=0),torch.cat(output_bw, dim=0)
   
    # @torch.no_grad()
    # def get_all_embed(self, data_loader: DataLoader,positions,gene_pos):
    #     output_gene = []
    #     output_exp = [] 
    #     output_weight = []
    #     output_mean = []
    #     for data in tqdm(data_loader):
    #         inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
    #         # inputs=torch.squeeze(data['ocrb_input'], dim=1)
    #         # inputs=inputs.to('cuda:1')
    #         output, _ = self.model.ocrb.forward(inputs,positions)
    #         output_gene.append(torch.squeeze(output[:,gene_pos,:], dim=1))
    #         output_weight.append(torch.squeeze(output[:,gene_pos,:], dim=1)*torch.unsqueeze(inputs[:, gene_pos], dim=1))
    #         binary_tensor = (inputs[:, gene_pos] != 0).to(torch.float32)
    #         output_exp.append(torch.squeeze(output[:,gene_pos,:], dim=1)*torch.unsqueeze(binary_tensor, dim=1))
    #         # output_weight.append(torch.sum(output*torch.unsqueeze(inputs, dim=-1),dim=1))
    #         output_mean.append(torch.mean(output*torch.unsqueeze(inputs, dim=-1),dim=1))
    #     return torch.cat(output_gene, dim=0) ,torch.cat(output_weight, dim=0),torch.cat(output_exp, dim=0),torch.cat(output_mean, dim=0)
    
    
    
    @torch.no_grad()
    def get_gene_embed(self, data_loader: DataLoader,positions,gene_pos):
        output_gene = [] 
        output_exp = [] 
        for data in tqdm(data_loader):
            inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
            output, _ = self.model.ocrb.forward(inputs,positions)
            output_gene.append(torch.squeeze(output[:,gene_pos,:], dim=1))
            
            binary_tensor = (inputs[:, gene_pos] != 0).to(torch.float32)
            output_exp.append(torch.squeeze(output[:,gene_pos,:], dim=1)*torch.unsqueeze(binary_tensor, dim=1))
        return torch.cat(output_gene, dim=0) ,torch.cat(output_exp, dim=0)

    @torch.no_grad()
    def FE(self, data_loader: DataLoader,positions,gene_pos):
        output_exp = [] 
        pca = PCA(n_components=1)
        for data in tqdm(data_loader):
            inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
            output, _ = self.model.ocrb.forward(inputs,positions) 
            binary_tensor = (inputs[:, gene_pos] != 0).to(torch.float32)
            output_exp.append(torch.squeeze(output[:,gene_pos,:], dim=1)*torch.unsqueeze(binary_tensor, dim=1))
        embeds=torch.cat(output_exp, dim=0)
        w= pca.fit_transform(embeds.cpu().numpy())
        return w
    
    @torch.no_grad()
    def get_attn(self, data_loader: DataLoader,positions):
        attn_weights = []
        for data in tqdm(data_loader):
            inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
            _, attn = self.model.ocrb.forward(inputs,positions)
            attn_weights.append(attn)
        return torch.cat(attn_weights, dim=0)
    
    # @torch.no_grad()
    # def get_attn(self, data_loader: DataLoader,positions,gene_pos):
    #     attn_weights = []
    #     for data in tqdm(data_loader):
    #         inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
    #         # inputs=torch.squeeze(data['ocrb_input'], dim=1)
    #         # inputs=inputs.to('cuda:1')
    #         _, attn = self.model.ocrb.forward(inputs,positions)
    #         attn_weights.append(attn)
    #     return torch.cat(attn_weights, dim=0)



    def test_result(self):
        return self.get_results(self.test_data)
        
    def train_result(self):
        return self.get_results(self.train_data)
    
    
    # @torch.no_grad()
    # def get_w_embed(self, data_loader: DataLoader,gene_pos,positions):
    #     outputs_1= [] 
    #     outputs_2= [] 
    #     outputs_3= []
    #     for data in tqdm(data_loader):
    #         inputs=torch.squeeze(data['ocrb_input'], dim=1).to(self.device)    
    #         output, _ = self.model.ocrb.forward(inputs,positions)
    #         gene_embedding=torch.squeeze(output[:,gene_pos,:], dim=1)*torch.unsqueeze(inputs[:, gene_pos], dim=1)
    #         outputs_1.append(gene_embedding)
    #         gene_embedding_2 = gene_embedding + torch.squeeze(output[:,gene_pos,:], dim=1)
    #         gaa_embedding=torch.mean(output*torch.unsqueeze(inputs, dim=-1)+output, dim=1)
    #         outputs_2.append(gaa_embedding)
    #         outputs_3.append(gene_embedding_2)
    #     return torch.cat(outputs_1, dim=0),torch.cat(outputs_2, dim=0),torch.cat(outputs_3, dim=0)
    
    