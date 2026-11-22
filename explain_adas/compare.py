import numpy as np
from scipy.stats import spearmanr


def rank_correlation(a, b):
    """Spearman rank correlation of two lists of region values."""
    result = spearmanr(a, b)
    return float(result.statistic)


def top_k_overlap(a, b, k=3):
    """How many of the k highest regions are in both lists (0 to k)."""
    top_a = set(np.argsort(a)[::-1][:k])
    top_b = set(np.argsort(b)[::-1][:k])
    return len(top_a & top_b)
