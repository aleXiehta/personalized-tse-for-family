import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import pandas as pd


class PTSEDataset(Dataset):
    def __init__(
            self, 
            csv_dir="pdns_training_set/train_aligned.csv",
            seg_len=None,
            sample_rate=16000,
        ):
        self.df = pd.read_csv(csv_dir)
        self.seg_len = int(seg_len * sample_rate)
        self.resampler = torchaudio.transforms.Resample(16000, sample_rate)
        self.sample_rate = sample_rate

    def load_wav(self, path):
        audio, sr = torchaudio.load(path)
        audio = self.resampler(audio)
        return audio
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        clean_wave, noisy_wave = map(self.load_wav, [sample["clean"], sample["noisy"]])
        # spk_emb = torch.from_numpy(np.load(sample["spk_emb"]))
        if self.seg_len is not None:
            # assert self.seg_len <= clean_wave.shape[-1]
            # assert self.seg_len <= noisy_wave.shape[-1]
            st = np.random.randint(0, clean_wave.shape[-1] - self.seg_len - self.sample_rate, (1,))[0]
            enrol_wave = clean_wave[..., -self.seg_len:]
            clean_wave = clean_wave[..., st:st + self.seg_len]
            noisy_wave = noisy_wave[..., st:st + self.seg_len]
        
        return {
            "clean_wave": clean_wave,
            "noisy_wave": noisy_wave,
            "enrol_wave": enrol_wave,
            # "spk_emb": spk_emb,
        }


class PTSE_FT_Dataset(Dataset):
    def __init__(
            self, 
            csv_dir="pdns_training_set/test_csv/test_attenuated_0.csv",
            finetune=True,
            seg_len=None,
            sample_rate=8000,
        ):
        self.df = pd.read_csv(csv_dir)
        if finetune:
            self.df = self.df[:720]
        else:
            self.df = self.df[720:]
        self.seg_len = int(seg_len * sample_rate)
        self.resampler = torchaudio.transforms.Resample(16000, sample_rate)
        self.sample_rate = sample_rate
        self.finetune = finetune

    def load_wav(self, path):
        audio, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            audio = self.resampler(audio)
        return audio
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        clean_wave, noisy_wave, pclean_wave = map(
            self.load_wav, 
            [sample["clean"], sample["noisy"], sample["pseudo_clean"]],
        )
        # spk_emb = torch.from_numpy(np.load(sample["spk_emb"]))
        if self.seg_len is not None and self.finetune:
            # assert self.seg_len <= clean_wave.shape[-1]
            # assert self.seg_len <= noisy_wave.shape[-1]
            st = np.random.randint(0, clean_wave.shape[-1] - self.seg_len - self.sample_rate, (1,))[0]
            enrol_wave = clean_wave[..., -self.seg_len:]
            clean_wave = clean_wave[..., st:st + self.seg_len]
            noisy_wave = noisy_wave[..., st:st + self.seg_len]
            pclean_wave = pclean_wave[..., st:st + self.seg_len]

        if not self.finetune:
            st = np.random.randint(0, clean_wave.shape[-1] - self.seg_len - self.sample_rate, (1,))[0]
            enrol_wave = clean_wave[..., -self.seg_len:]
        
        return {
            "clean_wave": clean_wave,
            "noisy_wave": noisy_wave,
            "enrol_wave": enrol_wave,
            "pclean_wave": pclean_wave,
            # "spk_emb": spk_emb,
        }