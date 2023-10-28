import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .random_noise import RandomNoiseSNR


def padding(audio, max_length):
    return F.pad(audio, (0, max_length - audio.shape[-1]))


def collate_fn_padd(data_list):
    """
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    """
    result = {}
    clean_audios = [i["clean_audio"] for i in data_list]
    noise_audios = [i["noise_audio"] for i in data_list]
    transcriptions = [i["transcription"] for i in data_list]
    max_length = max([i.shape[-1] for i in clean_audios])
    clean_audios = [padding(i, max_length) for i in clean_audios]
    noise_audios = [padding(i, max_length) for i in noise_audios]
    clean_audios = torch.stack(clean_audios, dim=0)
    noise_audios = torch.stack(noise_audios, dim=0)
    result["clean_audios"] = clean_audios
    result["noise_audios"] = noise_audios
    result["transcriptions"] = transcriptions
    return result


def make_loaders(common_voice, batch_size, desire_snr_db, max_length):
    print("Start loading datasets")
    train_dataset = RandomNoiseSNR(
        common_voice["train"], desire_snr_db=desire_snr_db, max_length=max_length
    )
    print("Train dataset loaded")
    test_dataset = RandomNoiseSNR(
        common_voice["test"], desire_snr_db=desire_snr_db, max_length=max_length
    )
    print("Test dataset loaded")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn_padd, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=collate_fn_padd, shuffle=False
    )
    return train_loader, test_loader
