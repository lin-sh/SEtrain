"""
single GPU version.
"""
import os
import torch
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from pesq import pesq
from send import send_email
from AECMOS_local.aecmos import AECMOSEstimator


class Trainer:
    def __init__(self, config, model, optimizer, loss_func,
                 train_dataloader, validation_dataloader, device, loss_type):
        self.config = config
        self.model = model
        self.loss_type = loss_type
        self.optimizer = optimizer
        self.aecmos = AECMOSEstimator(config['aecmos']['aecmos'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', factor=0.5, patience=5,verbose=True)
        self.loss_func = loss_func

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device

        ## training config
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']

        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
            
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')


        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)


        ## save the config
        with open(
            os.path.join(
                self.exp_path, 'config.toml'.format(datetime.now().strftime("%Y-%m-%d-%Hh%Mm"))), 'w') as f:

            toml.dump(config, f)

        self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0

        if self.resume:
            self._resume_checkpoint()

        self.sr = config['listener']['listener_sr']

        self.loss_func = self.loss_func.to(self.device)


    def _set_train_mode(self):
        self.model.train()

    def _set_eval_mode(self):
        self.model.eval()

    def _save_checkpoint(self, epoch, score):
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'model': self.model.state_dict(),
                      'score': score}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}.tar'))
        if score > self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score
            torch.save(self.state_dict_best,
                os.path.join(self.checkpoint_path,
                'best_model.tar'))  

    def _resume_checkpoint(self):
        best_checkpoint = os.path.join(self.checkpoint_path, 'best_model.tar')
        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.state_dict_best = {'epoch': checkpoint['epoch'],
                      'optimizer': checkpoint['optimizer'],
                      'model': checkpoint['model'],
                      'score': checkpoint['score']}.copy()
        self.best_score = checkpoint['score']


        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]
        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['model'])

    

    def _train_epoch(self, epoch):
        total_loss = 0
        self.train_dataloader = tqdm(self.train_dataloader, ncols=120)

        for step, (mixture, ref, target) in enumerate(self.train_dataloader, 1):
            mixture = mixture.to(self.device)
            ref = ref.to(self.device)
            target = target.to(self.device)  

            esti_tagt = self.model(mixture, ref)
            if self.loss_type == 'hybrid_CR':
                loss = self.loss_func(esti_tagt, target, mixture)
            else:
                loss = self.loss_func(esti_tagt, target)
            total_loss += loss.item()

            self.train_dataloader.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.train_dataloader.postfix = 'train_loss={:.3f}'.format(total_loss / step)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.optimizer.step()

        self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
        self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)


    @torch.no_grad()
    def _validation_epoch(self, epoch):
        total_loss = 0
        total_pesq_score = 0

        self.validation_dataloader = tqdm(self.validation_dataloader, ncols=132)
        for step, (mixture, ref, target) in enumerate(self.validation_dataloader, 1):
            mixture = mixture.to(self.device)
            ref = ref.to(self.device)
            target = target.to(self.device)  
            
            esti_tagt = self.model(mixture, ref)
            if self.loss_type == 'hybrid_CR':
                loss = self.loss_func(esti_tagt, target, mixture)
            else:
                loss = self.loss_func(esti_tagt, target)
            total_loss += loss.item()

            enhanced = torch.istft(esti_tagt[..., 0] + 1j*esti_tagt[..., 1], **self.config['FFT'], window = torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device))
            clean = torch.istft(target[..., 0] + 1j*target[..., 1], **self.config['FFT'], window = torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device))
            mixture = torch.istft(mixture[..., 0] + 1j*mixture[..., 1], **self.config['FFT'], window = torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device))
            ref = torch.istft(ref[..., 0] + 1j*ref[..., 1], **self.config['FFT'], window = torch.hann_window(self.config['FFT']['win_length']).pow(0.5).to(self.device))


            enhanced = enhanced.squeeze().cpu().numpy()
            clean = clean.squeeze().cpu().numpy()
            mixture = mixture.squeeze().cpu().numpy()
            ref = ref.squeeze().cpu().numpy()

            pesq_score = self.aecmos.run('dt', ref, mixture, enhanced)[0]
            # pesq_score = pesq(16000, clean, enhanced, 'wb')
            total_pesq_score += pesq_score

            if step <= 3:
                sf.write(os.path.join(self.sample_path,
                                    '{}_enhanced_epoch{}_pesq={:.3f}.wav'.format(step, epoch, pesq_score)),
                                    enhanced, 16000)
                sf.write(os.path.join(self.sample_path,
                                    '{}_clean.wav'.format(step)),
                                    clean, 16000)

            self.validation_dataloader.desc = 'validate[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch-1, datetime.now().strftime("%Y-%m-%d-%H:%M"))

            self.validation_dataloader.postfix = 'valid_loss={:.3f}, aecmos={:.4f}'.format(
                total_loss / step, total_pesq_score / step)

        self.writer.add_scalars(
            'val_loss', {'val_loss': total_loss / step,
                         'pesq': total_pesq_score / step}, epoch)

        return total_loss / step, total_pesq_score / step


    def train(self):
        timestamp_txt = os.path.join(self.exp_path, 'timestamp.txt')
        mode = 'a' if os.path.exists(timestamp_txt) else 'w'
        with open(timestamp_txt, mode) as f:
            f.write('[{}] start for {} epochs\n'.format(
                datetime.now().strftime("%Y-%m-%d-%H:%M"), self.epochs))

        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):

            self._set_train_mode()
            self._train_epoch(epoch)

            self._set_eval_mode()
            valid_loss, pesq_score = self._validation_epoch(epoch)

            self.scheduler.step(valid_loss)

            if epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch, pesq_score)
        torch.save(self.state_dict_best,
                os.path.join(self.checkpoint_path,
                'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))    

        print('------------Training for {} epochs has done!------------'.format(self.epochs))

        with open(timestamp_txt, 'a') as f:
            f.write('[{}] end\n'.format(datetime.now().strftime("%Y-%m-%d-%H:%M")))
        send_email(self.config['network_config']['node'] + " " + self.config['network_config']['des'], "您的模型已经成功训练完成！")
        