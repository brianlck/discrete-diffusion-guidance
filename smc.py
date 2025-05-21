import torch
import torch.nn.functional as F
import math

def full_resample_scheme(ln_w):
    categorical = torch.distributions.Categorical(probs=normalize_weights(ln_w))
    return categorical.sample((ln_w.shape[0],))


def partial_resample_scheme(samples, ln_w, count):
    count = int(count)
    ln_w = F.log_softmax(ln_w)
    id = torch.argsort(ln_w)
    assert count % 2 == 0
    half_count = count // 2
    samples = torch.concat((samples[id[-half_count:]], samples[id[:-half_count]]))
    ln_w = torch.concat((ln_w[id[-half_count:]], ln_w[id[:-half_count]]))
    replace_id = full_resample_scheme(ln_w[:count])
    norm_constant = torch.logsumexp(ln_w[:count], dim=0, keepdim=True)
    samples[:count] = samples[replace_id]
    ln_w[:count] = -math.log(count) + norm_constant
    return samples, ln_w

def compute_ess(w, dim=0):
    ess = (w.sum(dim=dim))**2 / torch.sum(w**2, dim=dim)
    return ess

def normalize_weights(log_weights, dim=0):
    return torch.exp(normalize_log_weights(log_weights, dim=dim))

def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - torch.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights

def compute_ess_from_log_w(log_w, dim=0):
    return compute_ess(normalize_weights(log_w, dim=dim), dim=dim)