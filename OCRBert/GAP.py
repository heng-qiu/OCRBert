import scanpy as sc
import pandas as pd
import pyensembl as en
from pyensembl import EnsemblRelease
from itertools import chain
import numpy as np

def get_gap(adata_gex, adata_atac, save_path, up_downstream=500000):
    genes_df=pd.DataFrame(index=adata_gex.var_names)
    peaks_df = pd.DataFrame(index=adata_atac.var_names)

    # 填充peaks_df 
    # ([index,chrom,start,end])
    split = adata_atac.var_names.str.split(r"[-]")        
    peaks_df["contig"] = split.map(lambda x: x[0])
    peaks_df["start"] = split.map(lambda x: x[1]).astype(int)
    peaks_df["end"] = split.map(lambda x: x[2]).astype(int)
    peaks_df['name']=peaks_df.index

    # 填充genes_df
    # ([index,chrom,start,end])
    
    # 获取基因组坐标
    genome = EnsemblRelease(98)
    chrom=[]
    chromStart=[]
    chromEnd=[]
    
    if 'gene_id' in adata_gex.var_keys():
        for i in adata_gex.var['gene_id']:
            gene = genome.gene_by_id(i) 
            if gene:
                # chrom.append(f"chr{gene.contig}")
                chrom.append(gene.contig)
                chromStart.append(gene.start)
                chromEnd.append(gene.end)
            else:
                print(f"Gene :{i} not found.")
                
    elif 'name' in adata_gex.var_keys():
        for i in adata_gex.var['name']:
            gene = genome.genes_by_name(i)[0] 
            if gene:
                # chrom.append(f"chr{gene.contig}")
                chrom.append(gene.contig)
                chromStart.append(gene.start)
                chromEnd.append(gene.end)
            else:
                print(f"Gene :{i} not found.")

    else:
        raise ValueError("Please confirm the query")

    chr_list=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
            '11', '12', '13', '14', '15', '16', '17', '18', '19', 
            '20', '21', '22', 'X', 'Y', 'MT']

    chrom_list=[]
    for i in chrom:
        if i in chr_list:
            chrom_list.append(f"chr{i}")
        else :
            chrom_list.append(i)
    chrom_series=pd.Series(chrom_list,index=adata_gex.var_names)
    chromStart_series=pd.Series(chromStart,index=adata_gex.var_names)
    chromEnd_series=pd.Series(chromEnd,index=adata_gex.var_names)

    genes_df["contig"] = chrom_series
    genes_df["start"] = chromStart_series.astype(int)
    genes_df["end"] = chromEnd_series.astype(int)
    genes_df['name']=genes_df.index
    # 匹配

    genes_df['contig'] = genes_df['contig'].astype('category')
    peaks_df['contig'] = peaks_df['contig'].astype('category')

    # 根据'contig'列拆分数据
    genes_grouped = genes_df.groupby('contig')
    peaks_grouped = peaks_df.groupby('contig')

    genes_dict = {}
    peaks_dict = {}

    for contig, group in genes_grouped:
        genes_dict[contig] = group

    for contig, group in peaks_grouped:
        peaks_dict[contig] = group
        
    # 初始化空列表来收集匹配的gene-peak pairs
    gene_peak_dict = {}
    neighborhood_size = up_downstream  # ±500kb

    # 遍历每个染色体的数据
    for contig, genes in genes_dict.items():
        if contig in peaks_dict:
            # print('染色体：',contig)
            peaks = peaks_dict[contig]
            
            # 提取基因的名称、起始和终止位置作为NumPy数组
            gene_names = genes['name'].values
            gene_starts = genes['start'].values
            gene_ends = genes['end'].values
            
            # 提取峰值的名称、起始和终止位置作为NumPy数组
            peak_names = peaks['name'].values
            peak_starts = peaks['start'].values
            peak_ends = peaks['end'].values
            
            upstream_starts = np.maximum(0, gene_starts - neighborhood_size)
            downstream_ends = gene_ends + neighborhood_size
            
            # 遍历基因数据
            for gene_name, gene_start, gene_end, upstream_start, downstream_end in zip(
                    gene_names, gene_starts, gene_ends, upstream_starts, downstream_ends):
                
                # 在峰值数据中查找落在邻近区域内的峰值
                upstream_mask = (peak_starts >= upstream_start) & (peak_ends < gene_start)
                downstream_mask = (peak_starts > gene_end) & (peak_ends <= downstream_end)
                
                # 找到匹配的峰值
                upstream_matching_peaks = peak_names[upstream_mask]
                downstream_matching_peaks = peak_names[downstream_mask]
                # 将匹配的gene-peak添加到列表
                
                # upstream_matching_peaks + [gene_name]+ downstream_matching_peaks
                merged_list = list(chain(upstream_matching_peaks, [gene_name], downstream_matching_peaks))
                gene_peak_dict[gene_name]=merged_list

    gene_peak_df=  pd.DataFrame(dict([(k, pd.Series(v)) for k, v in gene_peak_dict.items()]))
    gene_peak_df.to_csv(save_path+'/gene_peak_df.csv', index=False)


if __name__ == "__main__":
    adata = sc.read('/home/zfeng/ssr/various_model/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad')
    adata_gex = adata[:,adata.var['feature_types']=='GEX'].copy()
    adata_atac = adata[:,adata.var['feature_types']=='ATAC'].copy() 
    get_gap(adata_gex, adata_atac, save_path='/home/zfeng/ssr/genes and peaks')