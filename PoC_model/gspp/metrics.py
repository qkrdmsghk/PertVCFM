from tkinter import FALSE
from typing import Callable, Dict, List, Tuple
import numpy as np
import scanpy as sc
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score
import scipy
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from lightning import LightningModule
from collections import defaultdict
import anndata
from tqdm import tqdm
from loguru import logger
import gspp.constants as cs
from sklearn.metrics.pairwise import rbf_kernel


# Pre-cache means of control, perts
# Global dictionary to store mean control values
CELL_TYPE_MEAN_CONTROL: Dict[str, np.ndarray] = {}
PERT_MEANS: Dict[str, pd.DataFrame] = {}


# TODO, the gloabl mean should be phased-out, but before doing so having 1:1 numbers
# from control matched vs global control is interesting; hence leaving for the moment
def compute_mean_control(adata, cell_type):
    """calculates and caches the mean control for each batch

    Args:
        adata: anndata.AnnData with obs values of 'condition' where 'ctrl' marks controls,
            'batch' for experimental batches, and 'cell_line' for the biological context (cell type, line, etc..)
        cell_type: string by which to subset biological context

    Mutates:
        CELL_TYPE_MEAN_CONTROL: Dict[str, pd.core.frame.DataFrame)]
            {cell_type: df of control means, batch X gene}

    Returns:
        None
    """
    if cell_type not in CELL_TYPE_MEAN_CONTROL or CELL_TYPE_MEAN_CONTROL[cell_type][1].shape[1] != adata.shape[1]:
        print(f"Computing mean control for {cell_type} with {adata.shape[1]} genes")
        temp_adata = adata[adata.obs[cs.CELL_TYPE] == cell_type]
        # setup to group controls by batch
        ctrl_mask = temp_adata.obs[cs.CONDITION] == cs.CONTROL_LABEL
        tmpx = temp_adata[ctrl_mask].X
        if isinstance(tmpx, scipy.sparse._csr.csr_matrix):
            tmpx = tmpx.toarray()
        tmpx = pd.DataFrame(tmpx)
        ctrls = tmpx.mean(0)

        tmpx["group"] = temp_adata.obs.loc[ctrl_mask, cs.BATCH].values.astype(str)
        # take the control mean for each batch
        batch_ctrls = tmpx.groupby("group").apply(lambda x: x.iloc[:, :-1].mean(0))

        CELL_TYPE_MEAN_CONTROL[cell_type] = (ctrls, batch_ctrls)

def cache_perturbation_means(adata, cell_type):
    """setup df with index pert and mean expression for each pert
    pert names are in 'pert_raw' format, e.g. gene symbols and do not contail +ctrl, such that they can be easily
    matched to pert passed to metric functions

    Args:
        adata: anndata object, with obs columns 'condition' marking unique perturbations, 'control' (mask) and 'cell_type'
        cell_type: cell type to use from adata
    """

    if cell_type not in PERT_MEANS:
        temp_adata = adata[adata.obs[cs.CELL_TYPE] == cell_type].copy()
        temp_adata.obs["pert_raw"] = [
            "+".join([p for p in perts.split("+") if p != "ctrl"])
            for perts in temp_adata.obs.condition
        ]
        # work on a non-sparse copy of X
        newx = temp_adata.X.copy()
        if isinstance(newx, scipy.sparse._csr.csr_matrix):
            newx = newx.toarray()
        # we want the perturbation deltas vs the closest control, so we will center on batches (i.e. subtract mean)
        control_mask = temp_adata.obs[cs.CONTROL].astype(bool).values
        for batch in temp_adata.obs[cs.BATCH].unique():
            batch_mask = temp_adata.obs[cs.BATCH] == batch
            # mean center in place
            control_mean = newx[batch_mask & control_mask].mean(0)
            newx[batch_mask] -= control_mean
        temp_adata.X = newx  # set centered values as X

        # mean aggregate the perturbation deltas
        temp_adata = temp_adata[~control_mask]
        means_x = sc.get.aggregate(temp_adata, by="pert_raw", axis=0, func=["mean"])
        PERT_MEANS[cell_type] = pd.DataFrame(
            means_x.layers["mean"], index=means_x.obs["pert_raw"]
        )


# Step 1: Metric Function Registry
METRIC_REGISTRY: Dict[
    str, Callable[[np.ndarray, np.ndarray, str, np.ndarray], float]
] = {}

TRAINING_METRIC_LIST: List[str] = []
TESTING_METRIC_LIST: List[str] = []
SLOW_METRIC_LIST: List[str] = []


def register_metric(name: str, training: bool = False, testing: bool = False, slow: bool = False):
    """Decorator to register a metric function in the METRIC_REGISTRY."""

    def decorator(
        func: Callable[[np.ndarray, np.ndarray, str, np.ndarray], float],
        training=training,
        testing=testing,
        slow=slow,
    ):
        METRIC_REGISTRY[name] = func
        if training:
            TRAINING_METRIC_LIST.append(name)
        if testing:
            TESTING_METRIC_LIST.append(name)
        if slow:
            SLOW_METRIC_LIST.append(name)
        return func

    return decorator


@register_metric("pearson", training=True, slow=False)
def pearson(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    
    p = pearsonr(pred.mean(0), truth.mean(0))[0]
    if np.isnan(p):
        return 0
    return p



@register_metric("r2", training=False, slow=False)
def r2(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    p = r2_score(pred.mean(0), truth.mean(0))
    if np.isnan(p):
        return 0
    return p



@register_metric("edist", training=False, slow=False)
def edist(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    
    # # 1. 下采样逻辑 (对于 E-distance 同样重要，因为也是 O(N^2))
    # # 既然 E-distance 不需要算 gamma，就不需要为了 gamma 单独算一次，
    # # 但为了计算 xx, yy, xy 的均值，如果细胞太多，还是建议下采样。
    # n_samples = 2000
    # if pred.shape[0] > n_samples:
    #     idx_p = np.random.choice(pred.shape[0], n_samples, replace=False)
    #     pred_sub = pred[idx_p]
    # else:
    #     pred_sub = pred
        
    # if truth.shape[0] > n_samples:
    #     idx_t = np.random.choice(truth.shape[0], n_samples, replace=False)
    #     truth_sub = truth[idx_t]
    # else:
    #     truth_sub = truth

    # # 2. 计算成对欧氏距离矩阵
    # # 也就是 ||X - X'||, ||Y - Y'||, ||X - Y||
    # d_xx = euclidean_distances(pred_sub, pred_sub)
    # d_yy = euclidean_distances(truth_sub, truth_sub)
    # d_xy = euclidean_distances(pred_sub, truth_sub)
    
    # # 3. 计算 Energy Distance
    # # 公式: 2 * Mean(XY) - Mean(XX) - Mean(YY)
    # edist = 2 * d_xy.mean() - d_xx.mean() - d_yy.mean()

    sigma_X = pairwise_squeuclidean(truth, truth).mean()
    sigma_Y = pairwise_squeuclidean(pred, pred).mean()
    delta = pairwise_squeuclidean(truth, pred).mean()
    edist = 2 * delta - sigma_X - sigma_Y
    if np.isnan(edist):
        edist = 0

    return edist      


def pairwise_squeuclidean(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared euclidean distances."""
    return ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)

def maximum_mean_discrepancy(x: np.ndarray, y: np.ndarray, gamma: float = 1.0) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two distributions x and y."""
    xx = rbf_kernel(x, x, gamma=gamma)
    yy = rbf_kernel(y, y, gamma=gamma)
    xy = rbf_kernel(x, y, gamma=gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()

@register_metric("mmd", training=False, slow=False)
def mmd(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    
    gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [maximum_mean_discrepancy(truth, pred, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))

@register_metric("mse", training=False, slow=False)
def mse_metric(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    return mse(pred.mean(0), truth.mean(0))


@register_metric("mae", training=False, slow=False)
def mae_metric(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    return mae(pred.mean(0), truth.mean(0))


@register_metric("wmae_delta_std", training=False, slow=False)
def wmae_delta_std(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
    eps: float = 1e-8,
) -> float:

    '''
    weight mae of the perturbation delta vs the control delta, normalized by the control std
    '''
    mu_ctrl = ctrl.mean(0)
    std_ctrl = ctrl.std(0)
    delta_pred = (pred.mean(0) - mu_ctrl) / std_ctrl
    delta_truth = (truth.mean(0) - mu_ctrl) / std_ctrl
    
    w = np.abs(delta_truth) + eps
    
    wmae = np.sum(w * np.abs(delta_pred - delta_truth)) / np.sum(w)
    if np.isnan(wmae):
        wmae = 0
    return wmae

@register_metric("sign_delta", training=False, slow=False)
def sign_delta(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
    eps: float = 1e-8,
) -> float:

    '''
    sign of the perturbation delta vs the control delta
    '''
    delta_pred = pred.mean(0) - ctrl.mean(0)
    delta_truth = truth.mean(0) - ctrl.mean(0)
    
    sign = np.mean(np.sign(delta_pred) == np.sign(delta_truth))

    if np.isnan(sign):
        sign = 0
    return sign


def ndcg_score(scores: np.ndarray, relevance: np.ndarray, k: int = 10) -> float:
    order = np.argsort(scores)[::-1][:k]
    rel = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(rel * discounts)

    ideal_order = np.argsort(relevance)[::-1][:k]
    ideal_rel = relevance[ideal_order]
    idcg = np.sum(ideal_rel * discounts)
    if idcg == 0:
        return 0

    return dcg / idcg
    
    
@register_metric("NDCG_delta_500", training=False, slow=False)
def NDCG_delta_500(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
    k: int = 500,
) -> float:
    
    delta_pred = pred.mean(0) - ctrl.mean(0)
    scores = np.abs(delta_pred)

    de_idx = adata.uns["de_idx_cache"][pert]
    relevance = np.zeros_like(scores)
    relevance[de_idx] = 1

    ndcg = ndcg_score(scores, relevance, k=k)
    if np.isnan(ndcg):
        ndcg = 0
    return ndcg



@register_metric("auroc_per_gene", training=False, testing=False)

@register_metric("pearson_delta", training=True, testing=True, slow=False)
def pearson_delta(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:

    pred = (pred.mean(0) - ctrl.mean(0))
    truth = (truth.mean(0) - ctrl.mean(0))

    val = pearsonr(pred, truth)[0]
    if np.isnan(val):
        val = 0
    return val

@register_metric("spearman_delta", training=True, slow=False)
def spearman_delta(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:

    val = spearmanr(pred.mean(0) - ctrl.mean(0), truth.mean(0) - ctrl.mean(0))[0]
    if np.isnan(val):
        val = 0
    return val


@register_metric("r2_delta", training=False, slow=False)
def r2_delta(
    pred: np.ndarray,
    truth: np.ndarray,
    cell_type: str,
    ctrl: np.ndarray,
    adata: anndata.AnnData,
    pert: str,
    dose: str = "1+1",
) -> float:
    
    val = r2_score(truth.mean(0) - ctrl.mean(0), pred.mean(0) - ctrl.mean(0))
    if np.isnan(val):
        val = 0
    return val


class BreakdownAssayedMetric:
    def __init__(self, metric_name: str, *args, **kwargs):
        self.metric_name = metric_name

    def mask(self, cell_type, adata, pert, dose: str, **kwargs):
        raise NotImplementedError

    def __call__(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        cell_type: str,
        ctrl: np.ndarray,
        adata: anndata.AnnData,
        pert: str,
        dose: str = "1+1",
    ) -> float:
        mask = self.mask(cell_type, adata, pert, dose)
                
        if mask is None:
            return np.nan
        
        if mask.shape[0] < 2:
            return np.nan
        
        else:
            subpred = pred[:, mask]
            subtruth = truth[:, mask]
            subctrl = ctrl[:, mask]
            return METRIC_REGISTRY[self.metric_name](
                subpred, subtruth, cell_type, subctrl, adata, pert, dose
            )

class PharosBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, kl: int, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)
        self.kl = kl

    def mask(self, cell_type, adata, pert, dose: str):
        return adata.uns["pharos"] == self.kl


# slight special handling for pharos breakdowns
pharoses = {
    f"pearson_delta_kl{kl}": PharosBreakdown("pearson_delta", kl) for kl in range(4)
}
for key, val in pharoses.items():
    register_metric(key, slow=False)(val)
import os

def cache_de_idx(adata):
    """setup {pert: de_idx} cache if not present"""
    if "de_idx_cache" not in adata.uns:
        res = {}
        gmap = {gene: i for i, gene in enumerate(adata.var_names)}
        
        for pert in adata.uns["rank_genes_groups_cov_all"]:
            
            res[pert+'+ctrl'] = np.array(
                [
                    gmap[gene]
                    for gene in adata.uns["rank_genes_groups_cov_all"][pert]
                    if gene in gmap
                ]
            )
        res[cs.CONTROL_LABEL] = np.array([-1] * len(res[pert+'+ctrl']))
        adata.uns["de_idx_cache"] = res



class DEBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, num_de_genes: int, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)
        self.num_de_genes = num_de_genes

    def mask(self, cell_type, adata, pert, dose: str):
        cache_de_idx(adata)  # setup cache if not present
        # perts = f"{cell_type}_{pert}_{dose}"
        perts = pert
        # print('perts', perts)
        # print('adata.uns["de_idx_cache"]', adata.uns["de_idx_cache"].keys())
        if perts in adata.uns["de_idx_cache"]:
            return adata.uns["de_idx_cache"][perts][: self.num_de_genes]
        else:
            return None


# slight special handling for DE breakdowns
de_breakdowns = {}
for methods in ['pearson', 'pearson_delta', 'spearman_delta', 'wmae_delta_std', 'sign_delta', 'mmd', 'edist']:
    for num_de_genes in [20, 50, 100, 500]:
        de_breakdowns[f"{methods}_de{num_de_genes}"] = DEBreakdown(methods, num_de_genes)
for key, val in de_breakdowns.items():
    register_metric(key, slow=True)(val)



class RetrievalMetric:
    def __init__(self, subset=False, *args, **kwargs):
        self.subset = subset

    def __call__(
        self,
        pred: np.ndarray,
        truth: np.ndarray,
        cell_type: str,
        ctrl: np.ndarray,
        adata: anndata.AnnData,
        pert: str,
        dose: str = "1+1",
    ) -> float:
        """relative retrieval rank of true reference match for pert, to pred of pert, higher is better, uses pearson_delta"""
        pert = "+".join([p for p in pert.split("+") if p != "ctrl"])

        truth_deltas = PERT_MEANS[cell_type]
        if self.subset:
            state = np.random.get_state()
            np.random.seed(0)
            n = min(truth_deltas.shape[0], 100)
            samp = np.random.choice(
                list(range(len(truth_deltas.index))), n, replace=False
            )
            np.random.set_state(state)
            truth_index = truth_deltas.index[samp].tolist()
            if pert not in truth_index:
                truth_index.append(pert)
            truth_deltas = truth_deltas.loc[truth_index]

        sims = []
        assert pert in truth_deltas.index
        # calculate similarity of pred to everything in truth_means
        for item in truth_deltas.index:
            sims.append(
                (pearsonr((pred - ctrl).mean(0), truth_deltas.loc[item])[0], item)
            )

        sims = sorted(sims)  # sort by similarity
        rank = [sim[1] for sim in sims].index(pert)  # find rank of query pert
        return rank / len(sims)


register_metric("normed_retrieval", slow=False)(RetrievalMetric(subset=False))
register_metric("fast_retrieval", testing=True, slow=False)(RetrievalMetric(subset=True))


class GraphConnectivityBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)

    def mask(self, cell_type, adata, pert, dose: str):
        pass


class GraphProximityBreakdown(BreakdownAssayedMetric):
    def __init__(self, metric_name: str, *args, **kwargs):
        super().__init__(metric_name, *args, **kwargs)

    def mask(self, cell_type, adata, pert, dose: str):
        pass



def metrics_calculation(
    results: dict, metrics_list: List[str], adata, match_cntr: bool = True
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """
    Generates generic metrics for a given results.
    Args:
            results (dict): {
                    pred: predictions,
                    truth: ground truth,
                    pred_cat: condition categories,
                    pred_de: differentially expressed genes,
                    truth_de: differentially expressed genes,
                    cell_type: cell types,
                    pert_cat: perturbation categories,
                }
            metrics_list (list of str): List of metric names to compute.
            adata: AnnData object containing the dataset.
            match_cntr: bool, whether to match control
    Returns:
            metrics (dict): {metric_name: metric_value}
            metrics_pert (dict): {perturbation: {metric_name: metric_value}}
    """
    metrics: Dict[str, List[float]] = {}
    metrics_pert: Dict[str, Dict[str, float]] = {}
    metrics = defaultdict(list)
    metrics_pert = defaultdict(lambda: defaultdict(list))

    ct_mask_dict: Dict[str, np.ndarray] = {
        cell_type: results[cs.RES_CELL_TYPE] == cell_type
        for cell_type in np.unique(results[cs.RES_CELL_TYPE])
    }

    for metric_name in metrics_list:
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{metric_name}' is not registered.")
        metric_fn = METRIC_REGISTRY[metric_name]
        # logger.info(f"running {metric_name}")


        if metric_name == "auroc_per_gene":
            # print("auroc_per_gene is not supported for flow-in-data-space")
            # logger.info("loading the test keys in perturbQA dataset")
            import sys
            sys.path.append('/home/yinhua/CellPertQA')  # 把上级目录加入 sys.path
            from PerturbQA.perturbqa import load_de, auc_per_gene
            # data_name = input("enter the dataset name: k562, rpe1, hepg2, jurkat, k562_set: ")
            data_name = results[cs.RES_CELL_TYPE][0].lower()
            data_de = load_de(data_name)
            test_keys = [(x["pert"], x["gene"]) for x in data_de["test"]]
            # logger.info(f"len(test_keys): {len(test_keys)}")
            
            gene_names = adata.var.gene_name.values
            pred_de = []
            truth_de = []
            test_keys_ = []
            label_ = []
            control_mean = adata[adata.obs.control==True].X.toarray().mean(axis=0)

            for i, (p, g) in enumerate(test_keys):
                if p+'+ctrl' in results[cs.RES_PERT_CAT]:
                    p_idx = np.where(results[cs.RES_PERT_CAT] == p+'+ctrl')[0]
                    pred_cells = results[cs.RES_PRED][p_idx].mean(axis=0)
                    truth_cells = results[cs.RES_TRUTH][p_idx].mean(axis=0)
                    if g in gene_names:
                        g_idx = np.where(gene_names == g)[0]
                        pred_de.append(np.abs(pred_cells[g_idx] - control_mean[g_idx])[0])
                        truth_de.append(np.abs(truth_cells[g_idx] - control_mean[g_idx])[0])
                        label_.append(data_de["test"][i]["label"])
                        test_keys_.append((p, g))

            pred_de = np.array(pred_de)
            truth_de = np.array(truth_de)
            label_ = np.array(label_)
            if pred_de.shape[0] == 0 or truth_de.shape[0] == 0 or np.isnan(pred_de).any() or np.isnan(truth_de).any():
                pcc_delta_de = 0
                auc = 0
            else:
                pcc_delta_de = pearsonr(pred_de, truth_de)[0]
                auc = auc_per_gene(test_keys_, pred_de, label_)
            metrics[metric_name] = auc
            metrics["pcc_delta_de"] = pcc_delta_de
        else:
            
            for pert in tqdm(np.unique(results[cs.RES_PERT_CAT]), ncols=50, desc=metric_name):
                for dose in np.unique(results[cs.RES_DOSE_CAT]):
                    p_d_mask = np.logical_and(
                        results[cs.RES_PERT_CAT] == pert, results[cs.RES_DOSE_CAT] == dose
                    )
                    if np.sum(p_d_mask) == 0:
                        continue                
                    for cell_type in np.unique(results[cs.RES_CELL_TYPE]):
                        ct_mask = ct_mask_dict[cell_type]
                        p_d_ct_mask = p_d_mask & ct_mask
                        # If this cell type + perturbation combination has no data, skip
                        if np.sum(p_d_ct_mask) == 0:
                            continue

                        # Compute mean of control and each pert if not already cached
                        compute_mean_control(adata, cell_type)
                        cache_perturbation_means(adata, cell_type)
                        
                        if match_cntr:
                            control = (
                                CELL_TYPE_MEAN_CONTROL[cell_type][1]
                                .loc[results[cs.RES_EXP_BATCH][p_d_ct_mask].astype(str), :]
                                .to_numpy()
                            )
                        else:
                            print('no matching control')
                            control = (
                                CELL_TYPE_MEAN_CONTROL[cell_type][0]
                                .to_numpy()
                                .reshape(1, -1)
                            )

                        assert (
                            control.shape[0] > 0
                        ), f"No control found for {cell_type} x {pert} x {dose}"


                        # Calculate primary metric
                        value = metric_fn(
                            results[cs.RES_PRED][p_d_ct_mask],
                            results[cs.RES_TRUTH][p_d_ct_mask],
                            cell_type,
                            control,
                            adata,
                            pert,
                            dose,
                        )

                        value = 0 if np.isnan(value) else value
                        metrics_pert[pert][metric_name].append(value)
                        metrics[metric_name].append(value)
                metrics_pert[pert][metric_name] = np.mean(metrics_pert[pert][metric_name])
            metrics[metric_name] = np.mean(metrics[metric_name])
        
    return metrics, metrics_pert



def metrics_calculation_cellflow(
    results: dict, 
    metrics_list: List[str], 
    adata, 
    match_cntr: bool = True
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, float]]]:
    """
    """    
    metrics: Dict[str, List[float]] = {}
    metrics_pert: Dict[str, Dict[str, float]] = {}
    metrics = defaultdict(list)
    metrics_pert = defaultdict(lambda: defaultdict(list))

    cell_type = 'K562'
    for metric_name in metrics_list:
        if metric_name not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{metric_name}' is not registered.")
        metric_fn = METRIC_REGISTRY[metric_name]
        for pert in tqdm(results, ncols=50, desc=metric_name):
            
            truth = adata[adata.obs.gene_name == pert].X.toarray()
            prediction = results[pert]
            if prediction.shape[0] > 500:
                # print(f'downsampling {pert} from {prediction.shape[0]} to 500 for each perturbation')
                prediction = prediction[np.random.choice(prediction.shape[0], size=500, replace=False), :]

            batch_id = adata[adata.obs.gene_name == pert].obs.batch.values.astype(str)

            # print('batch_id', batch_id.shape)
            # print('truth', truth.shape)
            # print('prediction', prediction.shape)

            if '+ctrl' not in pert:
                pert += '+ctrl'

            compute_mean_control(adata, cell_type=cell_type)
            cache_perturbation_means(adata, cell_type=cell_type)
            
            if match_cntr:
                control = (
                    CELL_TYPE_MEAN_CONTROL[cell_type][1]
                    .loc[batch_id.tolist(), :]
                    .to_numpy()
                )
            else:
                control = (
                    CELL_TYPE_MEAN_CONTROL[cell_type][0]
                    .to_numpy()
                    .reshape(1, -1)
                )

            assert (
                control.shape[0] > 0
            ), f"No control found for K562 x {pert}"

            control = (
                        CELL_TYPE_MEAN_CONTROL[cell_type][0]
                        .to_numpy()
                        .reshape(1, -1)
            )
            value = metric_fn(
                prediction,
                truth,
                cell_type=cell_type,
                ctrl=control,
                adata=adata,
                pert=pert,
                dose='1+1',
            )
            value = 0 if np.isnan(value) else value
            metrics_pert[pert][metric_name].append(value)
            metrics[metric_name].append(value)
            metrics_pert[pert][metric_name] = np.mean(metrics_pert[pert][metric_name])        
        metrics[metric_name] = np.mean(metrics[metric_name])        
                    
    return metrics, metrics_pert

def get_group_to_idx(vals):
    group_to_idx = defaultdict(list)
    for i, v in enumerate(vals):
        group_to_idx[v].append(i)
    return dict(group_to_idx)

from sklearn.metrics import roc_auc_score
def auc_per_gene(keys, pred, true, reduce=True):
    """
    AUC computed per gene

    @param (list) keys  [(pert, gene), ...]
    @param (list, np.ndarray) pred
    @param (list, np.ndarray) true
    """
    # sanitize
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)
    # group
    genes = [k[1] for k in keys]
    groups = get_group_to_idx(genes)
    metrics = []
    for group, idx in groups.items():
        ts = true[idx]
        ps = pred[idx]
        # skip all 0s or 1s (shouldn't be in the input)
        if sum(ts) == 0 or sum(ts) == len(ts):
            continue
        metrics.append(roc_auc_score(ts, ps))
    if reduce:
        metrics = np.mean(metrics).item()
    return metrics


def log_metrics(
    metrics: dict, metrics_pert: dict, stage: str, subgroup: dict, lm: LightningModule
):
    """
    Logs the metrics in the metrics dictionary
    Args:
            metrics (dict): {metric_name: metric_value}
            metrics_pert (dict): {perturbation: {metric_name: metric_value}}
            stage (str): Stage of the experiment (train, val, test)
            subgroup (dict): Dictionary containing the subgroup analysis
            lm (LightningModule): Lightning Module object
    """
    for metric_name, metric_value in metrics.items():
        log_key = f"{stage}_{metric_name}"
        logger.info(f"{log_key}: {metric_value}")
        lm.log(log_key, metric_value)

    # if stage == cs.TEST:
    #     subgroup_analysis = {}
    #     for name in subgroup["test_subgroup"].keys():
    #         subgroup_analysis[name] = {}
    #         for m in list(list(metrics_pert.values())[0].keys()):
    #             subgroup_analysis[name][m] = []

    #     for name, pert_list in subgroup["test_subgroup"].items():
    #         for pert in pert_list:
    #             for m, res in metrics_pert[pert].items():
    #                 if pert in metrics_pert.keys():
    #                     subgroup_analysis[name][m].append(res)

    #     for name, result in subgroup_analysis.items():
    #         for m in result.keys():
    #             subgroup_analysis[name][m] = np.mean(subgroup_analysis[name][m])
    #             lm.log("test_" + name + "_" + m, subgroup_analysis[name][m])


def compute_metrics(results, adata, metric_stage, match_cntr: bool = True):
    """
    Given results from a model run and the ground truth, and the stage as 'fast' or 'extended': compute metrics
    """
    if metric_stage == cs.FAST:
        metrics_list = TRAINING_METRIC_LIST
    elif metric_stage == cs.EXTENDED:
        metrics_list = TESTING_METRIC_LIST
    elif metric_stage == cs.SLOW:
        metrics_list = SLOW_METRIC_LIST
    else:
        raise ValueError(f"Invalid metric stage: {metric_stage}")

    metrics_list = list(set(metrics_list))
    if 'pert_cat' in results:
        metrics, metrics_pert = metrics_calculation(results, metrics_list, adata, match_cntr)
    else:
        print('evaluating cellflow results...')
        metrics, metrics_pert = metrics_calculation_cellflow(results, metrics_list, adata, match_cntr)
    return metrics, metrics_pert