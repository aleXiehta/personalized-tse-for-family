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
from datasets import PTSE_FT_Dataset
import pandas as pd

parser = argparse.ArgumentParser(description="Training of SepFormer")
parser.add_argument('--device', type=str)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--csv_dir', type=str)
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
    return torch.nn.functional.pad(y_hat, (0, best_k))[:, best_k:], best_sisdr

model = separator.from_hparams(
    source="speechbrain/resepformer-wsj02mix", 
)
model_st = separator.from_hparams(
    source="speechbrain/resepformer-wsj02mix", 
)
ckpt = torch.load(args.ckpt_path)
ckpt_st = torch.load("exp/student/ckpt/epoch=99-step=225000.ckpt")
state_dict = dict()
state_dict_st = dict()
for old_k, v in ckpt["state_dict"].items():
    new_k = old_k[6:]
    state_dict[new_k] = ckpt["state_dict"][old_k]
for old_k, v in ckpt_st["state_dict"].items():
    new_k = old_k[6:]
    state_dict_st[new_k] = ckpt_st["state_dict"][old_k]

model.load_state_dict(state_dict)
model.cuda()
model.eval()
model_st.load_state_dict(state_dict_st)
model_st.cuda()
model_st.eval()

model_th = separator.from_hparams(
    source="speechbrain/sepformer-wham16k-enhancement", 
)
model_th.cuda()
model_th.eval()

test_set = PTSE_FT_Dataset(
    csv_dir=args.csv_dir,
    seg_len=1,
    sample_rate=8000,
    finetune=False,
)
test_set.df.reset_index(inplace=True)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print(f"Create {args.save_dir}")

sisdr = ScaleInvariantSignalDistortionRatio()
sisdr.cuda()
i = 0
all_sisdr = {
    "n2c": [],
    "pc2c": [],
    "st2c": [],
    "st2pc": [],
    "b2c": [],
    "b2pc": [],
}
for batch in (pbar := tqdm(test_set)):
    y, py, x, e = batch["clean_wave"], batch["pclean_wave"], batch["noisy_wave"], batch["enrol_wave"]
    y, py, x, e = y.cuda(), py.cuda(), x.cuda(), e.cuda()
    with torch.no_grad():
        y_hat = model(torch.cat([e, x], dim=-1).squeeze(1))
        y_hat_st = model_st(torch.cat([e, x], dim=-1).squeeze(1))
        y_hat_th = model_th(torch.cat([e, x], dim=-1).squeeze(1))
        y_hat = y_hat[..., 0]
        y_hat = y_hat[..., 8000:]
        y_hat_st = y_hat_st[..., 0]
        y_hat_st = y_hat_st[..., 8000:]
        y_hat_th = y_hat_th[..., 0]
        y_hat_th = y_hat_th[..., 8000:]

        pc2c = sisdr(y_hat_th, y).item()
        st2c = sisdr(y_hat, y).item()
        st2pc = sisdr(y_hat, y_hat_th).item()
        b2c = sisdr(y_hat_st, y).item()
        b2pc = sisdr(y_hat_st, y_hat_th).item()

        # _, st2pc = optimal_shift(y_hat, py, sisdr, 300)
        n2c = sisdr(x, y).item()
        # y_hat = shift(y_hat, off)
        
    all_sisdr["n2c"].append(n2c)
    all_sisdr["pc2c"].append(pc2c)
    all_sisdr["st2c"].append(st2c)
    all_sisdr["st2pc"].append(st2pc)
    all_sisdr["b2c"].append(b2c)
    all_sisdr["b2pc"].append(b2pc)
    # break

    noisy_path = test_set.df["noisy"][i]
    file_name = noisy_path.split("/")[-1]
    save_path = os.path.join(args.save_dir, file_name)
    torchaudio.save(save_path, y_hat.cpu(), sample_rate=8000)
    pbar.set_description(f"(enh -> cln): {st2c:.4}; (enh -> pcln): {st2pc:.4}; (stu -> cln): {b2c:.4}; (pcln -> cln): {pc2c:.4}; (nsy -> cln) {n2c:.4}")
    i += 1

df = pd.DataFrame.from_dict(all_sisdr)
df.to_csv(os.path.join(args.save_dir, "results.csv"))