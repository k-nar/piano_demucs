import torch
import tqdm
from torch import Tensor
import torchaudio

#DATASET_PATH = "F:/piano_restauration"
DATASET_PATH = "/Users/alex/dev/dataset/piano_restauration"


from torchaudio import functional as F
from torchaudio import transforms as T
import os

from torch_audiomentations import *
#from audiomentations import ClippingDistortion
# from audiomentations import *
import random
from audiomentations.core.transforms_interface import BaseWaveformTransform

SAMPLE_RATE = 44100
NB_SAMPLES_PER_FILE = 529200


class PianoDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir=DATASET_PATH + "/data", seq_len=NB_SAMPLES_PER_FILE, train = True, mono=False):
        self.wav_dir = wav_dir
        csv_file = "train.csv" if train else "validate.csv"
        self.wav_files = []
        with open(os.path.join(wav_dir, csv_file), 'r') as f:
            for line in f:
                self.wav_files.append(line.strip())
        self.seq_len = seq_len
        self.mono = mono

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, index):
        wav_file = self.wav_files[index]
        wav_path = os.path.join(self.wav_dir, wav_file)
        try:
            wav, sr = torchaudio.load(wav_path)
        except:
            print("error loading", wav_path)
            return None
        if self.seq_len < NB_SAMPLES_PER_FILE:
            start = random.randint(0, NB_SAMPLES_PER_FILE - self.seq_len)
            wav = wav[:, start:start + self.seq_len]
        if self.mono:
            wav = torch.mean(wav, dim=0, keepdim=True)
        return wav

    def __len__(self):
        return len(self.wav_files)



import time


def lp_degradation_(samples: Tensor):
    nb_cliped_samples = int(samples.shape[0] * 0.2)  # last 20% of the batch are clipped
    samples_to_clip = samples[-nb_cliped_samples:, ...]
    samples_to_clip = ClippingDistortionGPU(p=1)(samples_to_clip, sample_rate=SAMPLE_RATE)
    samples[-nb_cliped_samples:, ...] = samples_to_clip

    # last 90% of the batch
    nb_samples_transform = int(samples.shape[0] * 0.9)
    samples_transformed = samples[-nb_samples_transform:, ...]
    vinyl_noise = AddBackgroundNoise(background_paths=DATASET_PATH + "/augment/noise/vinyl", min_snr_in_db=3.0,
                                     max_snr_in_db=30.0, p=0.5, sample_rate=SAMPLE_RATE)
    #other_noise = AddBackgroundNoise(background_paths=DATASET_PATH + "/augment/noise/other", min_snr_in_db=1.0,
    #                                 max_snr_in_db=20.0, p=0.33, sample_rate=SAMPLE_RATE)
    bandstop = BandStopFilter(p=0.2, min_bandwidth_fraction=0.05, max_bandwidth_fraction=1.0, sample_rate=SAMPLE_RATE)
    reverb = ApplyImpulseResponse(ir_paths=DATASET_PATH + "/augment/reverb", p=0.30, sample_rate=SAMPLE_RATE)
    lowpass = LowPassFilter(p=0.7, min_cutoff_freq=1000, max_cutoff_freq=16000, sample_rate=SAMPLE_RATE)
    colored_noise = AddColoredNoise(p=0.3, sample_rate=SAMPLE_RATE)
    white_noise = AddColoredNoise(p=0.9, min_f_decay=0, sample_rate=SAMPLE_RATE)
    transforms = [bandstop, reverb, lowpass, vinyl_noise, colored_noise, white_noise]
    random.shuffle(transforms)
    for transform in transforms:
        samples_transformed = transform(samples_transformed, sample_rate=SAMPLE_RATE)

    samples[-nb_samples_transform:, ...] = samples_transformed
    return samples





class ClippingDistortionGPU:
    supports_multichannel = True

    def __init__(
            self,
            min_percentile_threshold: int = 0,
            max_percentile_threshold: int = 40,
            p: float = 0.5,
    ):
        """
        :param min_percentile_threshold: int, A lower bound on the total percent of samples that
            will be clipped
        :param max_percentile_threshold: int, An upper bound on the total percent of samples that
            will be clipped
        :param p: The probability of applying this transform
        """
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": None}
        self.are_parameters_frozen = False
        assert min_percentile_threshold <= max_percentile_threshold
        assert 0 <= min_percentile_threshold <= 100
        assert 0 <= max_percentile_threshold <= 100
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold
        self.randomize_parameters()

    def randomize_parameters(self):
        self.parameters["should_apply"] = random.random() < self.p
        if self.parameters["should_apply"]:
            self.parameters["percentile_threshold"] = random.randint(
                self.min_percentile_threshold, self.max_percentile_threshold
            )

    def __call__(self, samples: Tensor, sample_rate: int):
        lower_percentile_threshold = int(self.parameters["percentile_threshold"] / 2)
        lower_threshold = torch.quantile(samples, lower_percentile_threshold / 100)
        upper_threshold = torch.quantile(samples, 1 - lower_percentile_threshold / 100)
        samples = torch.clamp(samples, lower_threshold, upper_threshold)
        return samples


import time
import random

def get_piano_sample():
    dataset = PianoDataset()
    id = random.randint(0, len(dataset))
    return dataset[id]



def simple_auto_encoder(samples: Tensor):  # (BS, 2, 529200)
    bs, _, nb = samples.size()
    # turn to mono
    sample = torch.mean(samples, 1)
    # split into 1080 samples of 490
    sample = samples.reshape(bs, 1080, 490)


def create_csv(path=DATASET_PATH + "/data"):
    # list all files in a directory take 100 at random to create the validate.csv the others are in train.csv
    files = os.listdir(path)
    random.shuffle(files)
    nb_validate = 1024
    validate_files = files[:nb_validate]
    train_files = files[nb_validate:]
    with open(os.path.join(path, "validate.csv"), "w") as f:
        for file in validate_files:
            f.write(file + "\n")
    with open(os.path.join(path, "train.csv"), "w") as f:
        for file in train_files:
            f.write(file + "\n")


def play_sound(sound: Tensor):
    torchaudio.io.play_audio(sound.T, 44100)


if __name__ == '__main__':
    #test dataset
    dataset = PianoDataset(mono=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    clean = next(iter(dataloader))
    noisy = clean.clone()
    noisy = lp_degradation_(noisy)
    # save clean and noisy sound in test folder
    for i in range(len(clean)):
        torchaudio.save(f"/Users/alex/dev/demucs/test/{i}_clean.wav", clean[i], 44100)
        torchaudio.save(f"/Users/alex/dev/demucs/test/{i}_noisy.wav", noisy[i], 44100)

#   dataset = NoisyWavDataset()
#   dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
#   bs = next(iter(dataloader))
#   test = lp_degradation(bs, torch.device('mps'))
# #  play_sound_vlc(bs[0])
#  # time.sleep(0.5)
#   play_sound_vlc(test[0])


