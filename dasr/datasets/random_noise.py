import torch
from torch.utils.data import Dataset


class RandomNoiseSNR(Dataset):
    """
    Dataset that add random noise to common voice
    """

    def __init__(self, common_voice_subset, desire_snr_db=10, max_length=False):
        super().__init__()
        self.audio = common_voice_subset["audio"]
        self.transcriptions = common_voice_subset["sentence"]
        self.max_length = max_length
        self.desire_snr_db = desire_snr_db

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, idx):
        data = {}
        clean_audio = self.audio[idx]["array"]
        clean_audio = torch.Tensor(clean_audio)
        if self.max_length:
            clean_audio = clean_audio[0 : int(self.max_length)]
        noise_audio = self.noising(clean_audio, desire_snr_db=self.desire_snr_db)
        data["clean_audio"] = clean_audio
        data["noise_audio"] = noise_audio
        data["transcription"] = self.transcriptions[idx]
        return data

    def noising(self, wav, desire_snr_db):
        desire_snr_db = torch.Tensor([desire_snr_db])
        noise_samples = torch.randn(wav.shape)
        power_noise, power_wav = self.get_power(noise_samples), self.get_power(wav)

        noise_samples = noise_samples * torch.sqrt(power_wav / power_noise)
        snr = 10 ** (desire_snr_db / 10)

        noising_signal = torch.sqrt(snr) * wav + noise_samples
        noising_signal = noising_signal / torch.abs(noising_signal).max() * 0.4
        return noising_signal

    @staticmethod
    def get_power(signal):
        return (signal**2).sum()
