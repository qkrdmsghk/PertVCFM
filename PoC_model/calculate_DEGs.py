import scanpy as sc
import pandas as pd
import os
from tqdm import tqdm
import anndata as ad


import pandas as pd
import numpy as np

import argparse
import glob
import pickle

def create_predicted_anndata(results, adata, dataset):
    data_list = []  # 存储所有的细胞数据矩阵
    obs_list = []   # 存储所有的 obs DataFrame
        
    # --- A. 处理对照数据 ---
    num_control_cells = adata[adata.obs["gene_name"] == "non-targeting"].shape[0]
    data_list.append(adata[adata.obs["gene_name"] == "non-targeting"].X.toarray())
    
    control_obs = pd.DataFrame({
        'gene_name': ["non-targeting"] * num_control_cells,
    })
    obs_list.append(control_obs)
    obs_list.append(pd.DataFrame({'gene_name': results['pert_cat']}))
    data_list.append(results['pred'])
    
    # X: 最终的细胞表达矩阵
    X_matrix = np.vstack(data_list)
    # obs: 最终的观察值 DataFrame
    obs_df = pd.concat(obs_list, ignore_index=True)
    obs_df['gene_name'] = obs_df['gene_name'].apply(lambda x: x.replace('+ctrl', ''))
    
    var_df = pd.DataFrame(index=adata.var_names)
    
    # --- 4. 构建 AnnData 对象 ---
    
    adata = ad.AnnData(
        X=X_matrix,
        obs=obs_df,
        var=var_df
    )
    return adata

def cellflow_create_predicted_anndata(pert_dict, adata) -> ad.AnnData:
    
    data_list = []  # 存储所有的细胞数据矩阵
    obs_list = []   # 存储所有的 obs DataFrame
    
    # --- A. 处理对照数据 ---
    num_control_cells = adata[adata.obs["gene_name"] == "non-targeting"].shape[0]
    data_list.append(adata[adata.obs["gene_name"] == "non-targeting"].X.toarray())
    
    control_obs = pd.DataFrame({
        'gene_name': ["non-targeting"] * num_control_cells,
    })
    obs_list.append(control_obs)
    
    for condition_name, arrays in pert_dict.items():
        num_cells = arrays.shape[0]
        data_list.append(arrays)        
        current_obs = pd.DataFrame({
            'gene_name': [condition_name] * num_cells,
        })
        obs_list.append(current_obs)
            
    X_matrix = np.vstack(data_list)
    
    obs_df = pd.concat(obs_list, ignore_index=True)
    
    var_df = pd.DataFrame(index=adata.var_names)
    
    adata = ad.AnnData(
        X=X_matrix,
        obs=obs_df,
        var=var_df
    )
    return adata




def store_top_degs_per_perturbation_incremental(
    adata, 
    pert_key, 
    control_label, 
    n_top,
    abs_lfc_threshold,
    padj_threshold,
    save_dir
):
    # 1. 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建目录: {save_dir}")
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 获取所有扰动类型
    
    unique_perts = [p for p in adata.obs[pert_key].unique() if p != control_label]
    
    print(f"共 {len(unique_perts)} 个扰动。开始计算...")
    
    # 2. 循环计算
    
    for pert in tqdm(unique_perts):
        # --- 安全检查：文件名清理 ---
        #有些基因名可能有斜杠（如 A/B），作为文件名会报错，替换成下划线
        safe_name = str(pert).replace('/', '_')
        file_path = os.path.join(save_dir, f"{safe_name}.csv")
        
        # --- 断点续传逻辑 ---
        # 如果文件已经存在，说明之前算过了，直接跳过
        if os.path.exists(file_path):
            continue
        
        try:
            # --- 计算逻辑 ---
            # 策略：为了速度和内存，建议切片一个小数据计算，
            # 尤其是当你有 1000+ 组时，每次全量 rank_genes_groups 会越来越慢
            
            # 提取 当前组 和 对照组
            # 这一步虽然 copy 花点内存，但计算速度比在全量 adata 上跑 reference 快得多
            idx_mask = (adata.obs[pert_key] == pert) | (adata.obs[pert_key] == control_label)
            subset = adata[idx_mask].copy()
            
            sc.tl.rank_genes_groups(
                subset, 
                groupby=pert_key, 
                groups=[pert],
                reference=control_label,
                method='wilcoxon',
                key_added='temp_res'
            )
            
            # --- 提取结果 ---
            df = sc.get.rank_genes_groups_df(subset, group=pert, key='temp_res')
            
            # 筛选
            df ['abs_lfc'] = df['logfoldchanges'].abs()

            df = df[df['pvals_adj'] < padj_threshold] # 先滤掉不显著的
            df = df[df['abs_lfc'] > abs_lfc_threshold]
            
            # 如果筛选后空了（没有显著差异基因），存个空表防止报错，也作为"已计算"的标记
            if df.empty:
                df.to_csv(file_path, index=False)
                continue
                
            # 取 Top N
            cols = ['names', 'logfoldchanges', 'pvals_adj', 'scores']
            # 确保列存在
            cols = [c for c in cols if c in df.columns]
            top_df = df.nlargest(n_top, 'abs_lfc')[cols]
            top_df['pert_name'] = pert

            # --- 关键：立即写入硬盘 ---
            top_df.to_csv(file_path, index=False)
            
            # 释放内存
            del subset
            
        except Exception as e:
            print(f"\n[Error] 计算 {pert} 时出错: {e}")
            # 出错时不保存文件，这样下次重跑还会再试一次

    print(f"所有计算完成。结果保存在 {save_dir}/")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='K562_single_cell_line')
    parser.add_argument("--method", type=str, default='true_degs')
    parser.add_argument("--pert_key", type=str, default='gene_name')
    parser.add_argument("--control_label", type=str, default='non-targeting')
    parser.add_argument("--n_top", type=int, default=500)
    parser.add_argument("--abs_lfc_threshold", type=float, default=0.25)
    parser.add_argument("--padj_threshold", type=float, default=0.05)
    parser.add_argument("--save_dir", type=str, default='AAA_DEGs')

    args = parser.parse_args()
     
    dataset_name = args.dataset_name
    test_adata = ad.read_h5ad(f"cache/{dataset_name}/de_adata_test.h5ad")
    save_dir = os.path.join(args.save_dir, dataset_name, args.method)
    
    if args.method == 'true_degs':
         aa = test_adata
    else:
        pkl_files = glob.glob(f"AAA_results/{dataset_name}/{args.method}/*.pkl")
        results = pickle.load(open(pkl_files[0], "rb"))
        aa = create_predicted_anndata(results, test_adata, dataset_name)
        
    print(f"Calculating DEGs for {dataset_name} -- {args.method} will save to {save_dir} ...")
    store_top_degs_per_perturbation_incremental(
        aa, 
        pert_key=args.pert_key, 
        control_label=args.control_label,
        n_top=args.n_top,
        abs_lfc_threshold=args.abs_lfc_threshold,
        padj_threshold=args.padj_threshold,
        save_dir=save_dir
    )
    if args.method == 'true_degs':
        deg_idx_list = {}
        for deg in os.listdir(save_dir):
            df = pd.read_csv(os.path.join(save_dir, deg))
            pert = deg.split('.')[0]
            gene_idx = []
            for gene in df['names']:
                gene_idx.append(aa.var_names.get_loc(gene))
            deg_idx_list[pert] = np.array(gene_idx)
        pickle.dump(deg_idx_list, open(f'AAA_DEGs/{dataset_name}/true_degs_dict.pkl', 'wb'))