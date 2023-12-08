import os
import toml
import random
import torch
import pandas as pd
import soundfile as sf
from torch.utils import data


class MyDataset(data.Dataset):
    def __init__(self, train_folder, shuffle, num_tot, wav_len=0, n_fft=512, hop_length=256, win_length=512):
        super().__init__()
        ### We store the noisy-clean pairs in the same folder, and use CSV file to manage all the WAV files.
        self.file_name = pd.read_csv(train_folder, header=None)
        
        if shuffle:
          random.seed(7)
          self.file_name = self.file_name.sample(frac=1).reset_index(drop=True)
        
        if num_tot != 0:
          self.file_name = self.file_name[: num_tot]
        
        self.train_folder = train_folder
        self.wav_len = wav_len

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __getitem__(self, idx):       
        near, fs = sf.read(self.file_name.iloc[idx, 0], dtype="float32")
        ref, fs = sf.read(self.file_name.iloc[idx, 1], dtype="float32")
        clean, fs = sf.read(self.file_name.iloc[idx, 2], dtype="float32")
        

        near = torch.tensor(near)
        ref = torch.tensor(ref)
        clean = torch.tensor(clean)

        # near.len < 16000 add to 16000
        if near.shape[0] < fs * 10:
            # add zero
            near = torch.cat((near, torch.zeros(fs * 10 - near.shape[0])))
            ref = torch.cat((ref, torch.zeros(fs * 10 - ref.shape[0])))
            clean = torch.cat((clean, torch.zeros(fs * 10 - clean.shape[0])))

        if self.wav_len != 0:
            start = random.choice(range(len(clean) - self.wav_len * fs))
            near = near[start: start + self.wav_len*fs]
            ref = ref[start: start + self.wav_len*fs]
            clean = clean[start: start + self.wav_len*fs]

        near = torch.stft(near, self.n_fft, self.hop_length, self.win_length, torch.hann_window(self.win_length).pow(0.5), return_complex=False)
        ref = torch.stft(ref, self.n_fft, self.hop_length, self.win_length, torch.hann_window(self.win_length).pow(0.5), return_complex=False)
        clean = torch.stft(clean, self.n_fft, self.hop_length, self.win_length, torch.hann_window(self.win_length).pow(0.5), return_complex=False)

        return near, ref, clean
    
    def __len__(self):
        return len(self.file_name)


if __name__=='__main__':
    from tqdm import tqdm  
    config = toml.load('config.toml')

    device = torch.device('cuda')

    train_dataset = MyDataset(**config['train_dataset'], **config['FFT'])
    train_dataloader = data.DataLoader(train_dataset, **config['train_dataloader'])
    
    validation_dataset = MyDataset(**config['validation_dataset'], **config['FFT'])
    validation_dataloader = data.DataLoader(validation_dataset, **config['validation_dataloader'])

    print(len(train_dataloader), len(validation_dataloader))

    for near, ref, clean in tqdm(train_dataloader):
        print(near.shape, ref.shape, clean.shape)
        break
        # pass

    for near, ref, clean in tqdm(validation_dataloader):
        print(near.shape, ref.shape, clean.shape)
        # break
        # pass
 


