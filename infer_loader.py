import os
os.environ["CUDA_VISIBLE_DEVICES"]="" 

import toml
import torch
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from score_utils import sisnr
from omegaconf import OmegaConf
from datasets import MyDataset
from deepvqe_v1 import DeepVQE
from AECMOS_local.aecmos import AECMOSEstimator

@torch.no_grad()
def infer(cfg_yaml):

    # save_wavs = input('>>> Save wavs? (y/n) ')
    save_wavs = 'y'
    if save_wavs == 'y':
        # mark = input('>>> Please enter a tag for the saved wav names: ')
        mark = 'enhance'

    cfg_toml = toml.load(cfg_yaml.network.cfg_toml)
    cfg_toml['validation_dataset']['train_folder'] = '/root/aec/AEC-Challenge2022/datasets/synthetic/test_set.csv'
    cfg_toml['validation_dataset']['num_tot'] = 0         # all utterances
    cfg_toml['validation_dataset']['wav_len'] = 0         # full wav length
    cfg_toml['validation_dataloader']['batch_size'] = 1   # one utterence once

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    netout_folder = f'{cfg_yaml.path.exp_folder}'
    os.system(f'rm {netout_folder}/*.wav')
    os.makedirs(netout_folder, exist_ok=True)

    validation_dataset = MyDataset(**cfg_toml['validation_dataset'])
    validation_filename = validation_dataset.file_name
    
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, **cfg_toml['validation_dataloader'])

    ### load model
    model = DeepVQE()
    model.to(device)
    checkpoint = torch.load(cfg_yaml.network.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    ### compute SISNR, PESQ, and ESTOI
    INFO1 = []
    sisnr_score_total = 0
    pesq_score_total = 0
    estoi_score_total = 0
    scores0 = 0
    scores1 = 0
    aecmos = AECMOSEstimator()
    # INFO = pd.read_csv(os.path.join(cfg_yaml.path.csv_folder, 'INFO.csv'))
    for step, (mixture, ref, target) in enumerate(tqdm(validation_dataloader)):
            
        mixture = mixture.to(device)
        ref = ref.to(device)
        target = target.to(device)  
        
        estimate = model(mixture, ref)                    # [B, F, T, 2]

        enhanced = torch.istft(estimate[..., 0] + 1j*estimate[..., 1], **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5).to(device))
        clean = torch.istft(target[..., 0] + 1j*target[..., 1], **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5).to(device))

        out = enhanced.cpu().detach().numpy().squeeze()
        clean = clean.cpu().detach().numpy().squeeze()

        # out = torch.clamp(out, -1, 1)
        # out = out / out.max() * 0.5

        sisnr_score = sisnr(out, clean)
        pesq_score = pesq(16000, clean, out, 'wb')
        estoi_score = stoi(clean, out, 16000, extended=True)
        sisnr_score_total += sisnr_score
        pesq_score_total += pesq_score
        estoi_score_total += estoi_score

        save_name = ""
        ## save wavs
        if save_wavs == 'y':
            # save_name = "{}_{}_{:.2f}_{:.2f}_{:.2f}.wav".format(validation_filename[0][step], mark, sisnr_score, pesq_score, estoi_score)
            file_name = validation_filename[0][step].split('/')
            last_name = file_name[-1]
            last_name = last_name.split('_')[2:]
            last_name = '_'.join(last_name)
            save_name = "{}".format(last_name)
        
            sf.write(
                os.path.join(netout_folder, save_name), out, cfg_toml['listener']['listener_sr'])
        
        mixture = torch.istft(mixture[..., 0] + 1j*mixture[..., 1], **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5).to(device))
        ref = torch.istft(ref[..., 0] + 1j*ref[..., 1], **cfg_toml['FFT'], window=torch.hann_window(cfg_toml['FFT']['win_length']).pow(0.5).to(device))
        mixture = mixture.cpu().detach().numpy().squeeze()
        ref = ref.cpu().detach().numpy().squeeze()
        min_len = min(len(mixture), len(ref), len(out))
        mixture = mixture[:min_len]
        ref = ref[:min_len]
        out = out[:min_len]
        scores = aecmos.run('dt', ref, mixture, out)
        scores0 += scores[0]
        scores1 += scores[1]

        # save infos
        file_name = validation_filename[0][step]       
        INFO1.append([os.path.join(netout_folder, save_name), sisnr_score, pesq_score,  estoi_score])
    
    INFO1.insert(0, ['total', sisnr_score_total / len(validation_dataloader), pesq_score_total / len(validation_dataloader),  estoi_score_total / len(validation_dataloader)])
    INFO1.insert(1, ['AECMOS', scores0 / len(validation_dataloader), 'other',  scores1 / len(validation_dataloader)])
    INFO1 = pd.DataFrame(INFO1, columns=[cfg_yaml.network.cpt_name, 'sisnr', 'pesq', 'estoi'])
    INFO1.to_csv(os.path.join(cfg_yaml.path.csv_folder, 'INFO.csv'), index=None)


    # ### compute DNSMOS
    # os.chdir('DNSMOS')
    # out_dir = os.path.join(netout_folder, 'dnsmos_enhanced_p808.csv')
    # os.system(f'python dnsmos_local_p808.py -t {netout_folder} -o {out_dir}')
    
    
if __name__ == "__main__":
    cfg_yaml = OmegaConf.load('config.yaml')
    infer(cfg_yaml)
    
