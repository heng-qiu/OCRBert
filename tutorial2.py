import numpy as np
import scanpy as sc
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA

import sys
sys.path.append('/home/zfeng/ssr/genes and peaks/')
from OCRBert import OCRBTrainer

adata = sc.read('/home/zfeng/ssr/various_model/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')

adata.X=adata.layers['counts'].copy()
rna = adata[:, adata.var["feature_types"] == "GEX"].copy()
rna.X = rna.layers['counts'].copy()
sc.pp.normalize_total(rna, target_sum=1e4)
sc.pp.log1p(rna)
sc.pp.highly_variable_genes(rna, n_top_genes=2000, batch_key='batch')


rna_hvg = rna.var_names[rna.var['highly_variable']]
gene_peak_pd = pd.read_csv("/home/zfeng/ssr/genes and peaks/gene_peak_df.csv")
ocr_position_pd = pd.read_csv("/home/zfeng/ssr/genes and peaks/ocr_position.csv")

GAP_data=adata[:, gene_peak_pd['TMEM259'].dropna()].X
rna_data = rna[:, rna.var['highly_variable']].copy()
rna_data.X=rna_data.layers['counts'].copy()

# w=np.sum(rna_data.X.toarray(),axis=1)
# w=torch.from_numpy(w).to('cuda:0').unsqueeze(1)

vocab=np.unique(rna_data.X.data).astype(int)
mask_id = vocab[-1]+1

pos={}
for i in rna_hvg:
    if i in gene_peak_pd.columns:
        gene_pos =(gene_peak_pd[i].dropna() == i).idxmax()
        pos[i]=gene_pos

pos_code = {}
for i in rna_hvg:
    result=ocr_position_pd[i].dropna().str.split(r"[-]")
    start=result.map(lambda x: x[1]).astype(int).values
    end=result.map(lambda x: x[2]).astype(int).values
    coo=np.column_stack((start, end))
    pos_code[i]=coo


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
    
from sklearn.model_selection import train_test_split
train_dataset, test_dataset = train_test_split(GAP_data, test_size=0.2)

batch_size = 256
train_loader = DataLoader(OCRBDataset(train_dataset, vocab, mask_id),
                          batch_size=batch_size)
test_loader = DataLoader(OCRBDataset(test_dataset, vocab, mask_id),
                          batch_size=batch_size)
result=ocr_position_pd['TMEM259'].dropna().str.split(r"[-]")
start=result.map(lambda x: x[1]).astype(int).values
end=result.map(lambda x: x[2]).astype(int).values
positions =  np.column_stack((start, end))

model=OCRBTrainer(lr=1e-5,vocab_size= 7324, hidden=192,positions=positions,train_dataloader= train_loader, test_dataloader=test_loader,cuda_devices='cuda:0',warmup_steps=150)
for epoch in range(2):
    model.train(epoch)
    torch.cuda.empty_cache()
    model.test(epoch)
    torch.cuda.empty_cache()


batch_size=2048
t=0
# for i in rna_hvg[:500]:
for i in rna_hvg:
    if i in gene_peak_pd.columns:
        t=t+1
        print(f"第{t}个基因")
        GAP_data=adata[:, gene_peak_pd[i].dropna()].X
        # inputs = DataLoader(OCRBDataset(GAP_data, vocab, mask_id),
        #                       batch_size=batch_size)
        inputs = DataLoader(OCRBDataset(GAP_data),
                            batch_size=batch_size)
        # gene_pos =(gene_peak_pd[i].dropna() == i).idxmax()
        p=torch.from_numpy(pos_code[i]).to('cuda:0')
        weight= model.FE(inputs,p,pos[i])

        pd.DataFrame(weight).to_csv(f'./w/{i}.txt')
        torch.cuda.empty_cache()

        # if t > 600:
        #     break