import torch
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
from glob import glob
import os
from tqdm import tqdm
import IPython.display as ipd
from torchmetrics import ScaleInvariantSignalDistortionRatio
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Training of SepFormer")
parser.add_argument('--subset', type=str)
parser.add_argument('--device', type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device

def optimal_shift(y_hat, y, sisdr, max_shift=100):
    prev = -10000
    for k in range(max_shift):
        curr = sisdr(torch.nn.functional.pad(y_hat, (0, k))[:, k:], y)
        if curr > prev:
            prev = curr
            best_k = k
            best_sisdr = curr
    return best_k, best_sisdr.item()
    # return torch.nn.functional.pad(y_hat, (0, best_k))[:, best_k:], best_sisdr
    # return best_sisdr.item()

def shift(y_hat, k):
    return torch.nn.functional.pad(y_hat, (0, k))[:, k:]

model = separator.from_hparams(
    source="speechbrain/sepformer-wham16k-enhancement", 
    freeze_params=False,
)
ckpt = torch.load("exp/teacher/ckpt/epoch=98-step=222750.ckpt")
state_dict = dict()
for old_k, v in ckpt["state_dict"].items():
    new_k = old_k.replace("model.", "")
    state_dict[new_k] = ckpt["state_dict"][old_k]
model.load_state_dict(state_dict)
model.cuda()
model.eval()
sisdr = ScaleInvariantSignalDistortionRatio()
sisdr.cuda()

noisy_dir = f"../../pdns_training_set/attenuated/tt/noisy/{args.subset}/*.wav"
clean_root_dir = f"../../pdns_training_set/attenuated/tt/clean/{args.subset}"
pseudo_clean_root_dir = f"../../pdns_training_set/attenuated/tt/pseudo_clean/{args.subset}/"
pseudo_noisy_root_dir = f"../../pdns_training_set/attenuated/tt/pseudo_noisy/{args.subset}/"
if not os.path.exists(pseudo_clean_root_dir):
    os.makedirs(pseudo_clean_root_dir)
if not os.path.exists(pseudo_noisy_root_dir):
    os.makedirs(pseudo_noisy_root_dir)

resampler = torchaudio.transforms.Resample(16000, 8000)
noisy_paths = glob(noisy_dir)
for p in (pbar := tqdm(noisy_paths)):
    file_name = p.split("/")[-1]
    fileid = file_name.split("_")[3]
    clean_file_name = os.path.join(clean_root_dir, f"psn_clean_fileid_{fileid}.wav")
    x, sr = torchaudio.load(p)
    y, sr = torchaudio.load(clean_file_name)
    x = resampler(x)
    y = resampler(y)
    x = x.cuda()
    y = y.cuda()
    with torch.no_grad():
        y_hat = model(torch.cat([y[..., -8000:], x], dim=-1))
        y_hat.squeeze_(-1)
        y_hat = y_hat[..., 8000:]
        y_hat = y_hat / y_hat.abs().max()
        offset, pc2c = optimal_shift(y_hat, y, sisdr, 300)
        y_hat = shift(y_hat, offset)
        x = shift(x, offset)
    clean_save_path = os.path.join(pseudo_clean_root_dir, file_name)
    noisy_save_path = os.path.join(pseudo_noisy_root_dir, file_name)
    torchaudio.save(clean_save_path, y_hat.cpu(), sample_rate=8000)
    torchaudio.save(noisy_save_path, x.cpu(), sample_rate=8000)
    pbar.set_description(f"SISDR: {pc2c}, offset {offset}")
    # pbar.set_description(f"Saving to {save_path}")