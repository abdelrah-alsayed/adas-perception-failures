import numpy as np
import shap


def kernel_shap(model_fn, n_regions, nsamples, seed):
    """KernelSHAP values, one per region.

    model_fn gets a table of 0/1 values (rows = coalitions, 1 = region visible)
    and returns one number per row. All hidden is the baseline, all visible is
    the picture we explain. l1_reg=False keeps all regions (the library default
    would keep only the 10 most important and set the others to 0).
    """
    explainer = shap.KernelExplainer(model_fn, np.zeros((1, n_regions)))
    np.random.seed(seed)
    values = explainer.shap_values(np.ones((1, n_regions)), nsamples=nsamples,
                                   l1_reg=False, silent=True)
    return np.asarray(values).reshape(-1)
