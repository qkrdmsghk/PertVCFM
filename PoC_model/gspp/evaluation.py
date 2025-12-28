import torch
import numpy as np

import gspp.constants as cs
from tqdm import tqdm

def evaluate(loader, model, device, adata, id2pert, model_type=None):
    """
    Run model in inference mode using a given data loader
    """
    
    print('Staring get the predictions...')
    model.eval()
    pert_cat = []
    dose_cat = []
    pred = []
    truth = []
    cell_types = []
    experimental_batches = []
    results = {}

    pert_name_cache = {}
    # head obs is, e.g.
    # condition        control batch cell_type dose_val      condition_name
    # UBL5+ctrl        0       25    K562      1+1            K562_UBL5+ctrl_1+1
    # so the following makes a mapping from e.g. 'K562_UBL5+ctrl_1+1' to 'UBL5+ctrl' style naming
    for condition_name in np.unique(adata.obs[["condition_name"]]):
        pert_name_cache[condition_name] = (
            condition_name.split("_")[1],
            condition_name.split("_")[-1],
        )

    for batch in tqdm(loader):
        with torch.no_grad():
            # if model.no_pert:
            #     p = batch.control[:, : model.adata_output_dim]
            # else:
            p = model.sample_inference(
                batch.control, batch.pert_idxs, batch.p, batch.cell_types
            )
            p = p.cpu()
            t = batch.x.cpu()
            # control = batch.control.cpu()

            # mask_degs = batch.mask_degs.cpu()
            # non_mask_degs = ~mask_degs

            # 1. non deg = true values (ideal case)
            # p[non_mask_degs] = t[non_mask_degs]
            # 2. non deg = control values (match control)
            # p[non_mask_degs] = control[non_mask_degs]

            pred.append(p)
            truth.append(t)
            for itr, perts in enumerate(batch.pert_cond_names):
                pert_name = pert_name_cache[perts][0]
                dose = pert_name_cache[perts][1]
                pert_cat.append(pert_name)
                dose_cat.append(dose)
                cell_types.append(batch.cell_types[itr])
                experimental_batches.append(batch.experimental_batches[itr])

    # print(f"A total of {len(pert_cat)} perturbations were evaluated")
    results[cs.RES_PERT_CAT] = np.array(pert_cat)
    results[cs.RES_DOSE_CAT] = np.array(dose_cat)
    results[cs.RES_CELL_TYPE] = np.array(cell_types)
    results[cs.RES_PRED] = torch.cat(pred).numpy()
    results[cs.RES_TRUTH] = torch.cat(truth).numpy()
    results[cs.RES_TRUTH_BINARY] = None
    results[cs.RES_EXP_BATCH] = np.array(experimental_batches)


    return results
