import torch
import numpy as np
import librosa
import onnxruntime as ort
import argparse
import os
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--talk_type',
        type=str,
        default=None,
        help="Specify the scenario: 'st' (far-end single talk), 'nst' (near-end single talk), or 'dt' (double talk).")
    parser.add_argument('--model_path', type=str,
                        help="Specify the path to the onnx model provided")
    parser.add_argument('--lpb_path', type=str, required=True,
                        help="Specify the path to the lpb audio file")
    parser.add_argument('--mic_path', type=str, required=True,
                        help="Specify the path to the mic audio file")
    parser.add_argument('--enh_path', type=str, required=True,
                        help="Specify the path to the enh audio file")
    return parser.parse_args()


class AECMOSEstimator():

    def __init__(self):
        self.model_path = '/root/aec/SEtrain/AECMOS_local/Run_1663829550_Stage_0.onnx'
        self.max_len = 20
        self.hop_fraction = 0.5

        if 'Run_1663915512_Stage_0' in self.model_path:
            self.sampling_rate = 16000
            self.dft_size = 512
            self.transform = self._mel_transform
            self.need_scenario_marker = True
            self.hidden_size = (4, 1, 64)
        elif 'Run_1663829550_Stage_0' in self.model_path:
            self.sampling_rate = 16000
            self.dft_size = 512
            self.transform = self._mel_transform
            self.need_scenario_marker = False
            self.hidden_size = (4, 1, 64)
        elif 'Run_1668423760_Stage_0' in self.model_path:
            self.sampling_rate = 48000
            self.dft_size = 1536
            self.transform = self._mel_transform
            self.need_scenario_marker = True
            self.hidden_size = (4, 1, 96)
        else:
            ValueError, "Not a supported model."

    def _mel_transform(self, sample, sr):
        mel_spec = librosa.feature.melspectrogram(
            y=sample, sr=sr, n_fft=self.dft_size+1, hop_length=int(self.hop_fraction*self.dft_size), n_mels=160)
        mel_spec = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        return mel_spec.T

    def read_and_process_audio_files(self, lpb_path, mic_path, clip_path):
        lpb_sig, _ = librosa.load(lpb_path, sr=self.sampling_rate)
        mic_sig, _ = librosa.load(mic_path, sr=self.sampling_rate)
        enh_sig, _ = librosa.load(clip_path, sr=self.sampling_rate)

        # Make the clips the same length
        min_len = np.min([len(lpb_sig), len(mic_sig), len(enh_sig)])
        lpb_sig = lpb_sig[:min_len]
        mic_sig = mic_sig[:min_len]
        enh_sig = enh_sig[:min_len]
        return lpb_sig, mic_sig, enh_sig

    def run(self, talk_type, lpb_sig, mic_sig, enh_sig):
        assert len(lpb_sig) == len(mic_sig) == len(enh_sig)

        # cut segments if too long
        seg_nb_samples = self.max_len * self.sampling_rate
        if len(lpb_sig) >= seg_nb_samples:
            lpb_sig = lpb_sig[: seg_nb_samples]
            mic_sig = mic_sig[: seg_nb_samples]
            enh_sig = enh_sig[: seg_nb_samples]

        # feature transform
        lpb_sig = self.transform(lpb_sig, self.sampling_rate)
        mic_sig = self.transform(mic_sig, self.sampling_rate)
        enh_sig = self.transform(enh_sig, self.sampling_rate)

        # scenario marker
        if self.need_scenario_marker:
            assert talk_type in ['nst', 'st', 'dt']

            if talk_type == 'nst':
                ne_st = 1
                fe_st = 0
            elif talk_type == 'st':
                ne_st = 0
                fe_st = 1
            else:
                ne_st = 0
                fe_st = 0

            mic_sig = np.concatenate(
                (mic_sig, np.ones((20, mic_sig.shape[1])) * (1 - fe_st), np.zeros((20, mic_sig.shape[1]))), axis=0)
            lpb_sig = np.concatenate(
                (lpb_sig, np.ones((20, lpb_sig.shape[1])) * (1 - ne_st), np.zeros((20, lpb_sig.shape[1]))), axis=0)
            enh_sig = np.concatenate((enh_sig, np.ones(
                (20, enh_sig.shape[1])), np.zeros((20, enh_sig.shape[1]))), axis=0)

        # stack
        feats = np.stack((lpb_sig, mic_sig, enh_sig)).astype(np.float32)
        feats = np.expand_dims(feats, axis=0)

        # model_input = feats
        ort_session = ort.InferenceSession(self.model_path)
        input_name = ort_session.get_inputs()[0].name

        # GRU hidden layer shape is in h0
        with torch.no_grad():
            h0 = torch.zeros(self.hidden_size, dtype=torch.float32).detach().numpy()
        result = ort_session.run([], {input_name: feats, 'h0': h0})
        result = result[0]

        echo_mos = float(result[0])
        deg_mos = float(result[1])
        return echo_mos, deg_mos


if __name__ == "__main__":
    # args = parse_args()
    dataset_directory = "/root/aec/AEC-Challenge2022/datasets/synthetic"
    enhance_dir = "/root/aec/SEtrain/exp/exp_2023-11-29-17h30m/best_model_0101/enhanced/"
    aecmos = AECMOSEstimator()
    files = os.listdir(enhance_dir)
    scores0 = 0
    scores1 = 0
    for file_id in tqdm(files):  # 假设有 10000 个示例
        nearend_mic_signal_path = os.path.join(dataset_directory, f'nearend_mic_signal/nearend_mic_{file_id}')
        farend_speech_path = os.path.join(dataset_directory, f'farend_speech/farend_speech_{file_id}')
        enhance_path = os.path.join(enhance_dir, file_id)
        lpb_sig, mic_sig, enh_sig = aecmos.read_and_process_audio_files(
            farend_speech_path, nearend_mic_signal_path, enhance_path)
        scores = aecmos.run('dt', lpb_sig, mic_sig, enh_sig)
        scores0 += scores[0]
        scores1 += scores[1]
    print(f'The AECMOS echo score is {scores0 / len(files)}, and (other) degradation score is {scores1 / len(files)}.')
