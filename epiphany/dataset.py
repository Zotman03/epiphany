import torch
import h5py
import pickle
import numpy as np
import torch.utils.data

class HiCDiffusionDataset(torch.utils.data.Dataset):
    def __init__(self, seq_length=200, window_size=14000, chroms=['chr22'], data_dir="./Epiphany_dataset"):
        self.seq_length = seq_length
        self.window_size = window_size
        self.chroms = chroms

        with h5py.File(f"{data_dir}/GM12878_X.h5", "r") as f:
            self.X = {chrom: f[chrom][:] for chrom in f.keys()}

        # Load Hi-C data (2D maps)
        with open(f"{data_dir}/GM12878_y.pickle", "rb") as f:
            self.Y = pickle.load(f)

    def __len__(self):
        return len(self.chroms)

    def __getitem__(self, idx):
        chrom = self.chroms[idx]
        X_chipseq = torch.tensor(self.X[chrom][:self.seq_length, :], dtype=torch.float32)
        Y_target = torch.tensor(self.Y[chrom][:self.seq_length, :self.seq_length], dtype=torch.float32)
        Y_noisy = torch.nn.functional.avg_pool2d(Y_target.unsqueeze(0), kernel_size=4).squeeze(0)

        return X_chipseq, Y_noisy, Y_target
