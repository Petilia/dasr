import pandas as pd

def determ_valid_length(model, cfg):
    segment = cfg.facebook_denoiser.segment
    sample_rate = cfg.denoiser.sample_rate
    length = int(segment * sample_rate)
    length = model.valid_length(length)
    return length

def sum_list_dicts(stats):
    stats = pd.DataFrame(stats)
    stats = stats.mean().to_dict()
    return stats