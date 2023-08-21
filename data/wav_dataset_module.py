import torch

from typing import Optional
import tqdm
from torch import Tensor
import torchaudio
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform, EmptyPathException
from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.utils.dsp import calculate_rms
from torch_audiomentations.utils.file import find_audio_files_in_paths
from torch_audiomentations.utils.io import Audio
from torch_audiomentations.utils.fft import rfft, irfft
from math import ceil

DATASET_PATH = "F:/piano_restauration"
#DATASET_PATH = "/Users/alex/dev/dataset/piano_restauration"


from torchaudio import functional as F
from torchaudio import transforms as T
import os

from torch_audiomentations import *
#from audiomentations import ClippingDistortion
# from audiomentations import *
import random

SAMPLE_RATE = 44100
NB_SAMPLES_PER_FILE = 529200

import random
from pathlib import Path
from typing import Union, List, Optional

import torch
from torch import Tensor



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

def _gen_noise(f_decay, num_samples, sample_rate, device):
    """
    Generate colored noise with f_decay decay using torch.fft
    """
    noise = torch.normal(
        torch.tensor(0.0, device=device),
        torch.tensor(1.0, device=device),
        (sample_rate,),
        device=device,
    )
    spec = rfft(noise)
    mask = 1 / (
            torch.linspace(1, (sample_rate / 2) ** 0.5, spec.shape[0], device=device)
            ** f_decay
    )
    spec *= mask
    noise = Audio.rms_normalize(irfft(spec).unsqueeze(0)).squeeze()
    noise = torch.cat([noise] * int(ceil(num_samples / sample_rate)))
    return noise[:num_samples]

class AddBackgroundNoiseWithColoredNoise(BaseWaveformTransform):
    """
    Add background noise to the input audio.
    """

    supported_modes = {"per_batch", "per_example", "per_channel"}

    # Note: This transform has only partial support for multichannel audio. Noises that are not
    # mono get mixed down to mono before they are added to all channels in the input.
    supports_multichannel = True
    requires_sample_rate = True

    supports_target = True
    requires_target = False

    def __init__(
        self,
        background_paths: Union[List[Path], List[str], Path, str],
        min_snr_in_db: float = 3.0,
        max_snr_in_db: float = 30.0,
        mode: str = "per_example",
        p: float = 0.5,
        p_mode: str = None,
        sample_rate: int = None,
        target_rate: int = None,
        output_type: Optional[str] = None,
        add_noise_to_background: bool = True,
    ):
        """

        :param background_paths: Either a path to a folder with audio files or a list of paths
            to audio files.
        :param min_snr_in_db: minimum SNR in dB.
        :param max_snr_in_db: maximum SNR in dB.
        :param mode:
        :param p:
        :param p_mode:
        :param sample_rate:
        """

        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
            sample_rate=sample_rate,
            target_rate=target_rate,
            output_type=output_type,
        )

        # TODO: check that one can read audio files
        self.background_paths = find_audio_files_in_paths(background_paths)

        if sample_rate is not None:
            self.audio = Audio(sample_rate=sample_rate, mono=True)

        if len(self.background_paths) == 0:
            raise EmptyPathException("There are no supported audio files found.")

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if self.min_snr_in_db > self.max_snr_in_db:
            raise ValueError("min_snr_in_db must not be greater than max_snr_in_db")
        self.add_noise_to_background = add_noise_to_background

    def random_background(self, audio: Audio, target_num_samples: int) -> torch.Tensor:
        pieces = []

        # TODO: support repeat short samples instead of concatenating from different files

        missing_num_samples = target_num_samples
        while missing_num_samples > 0:
            background_path = random.choice(self.background_paths)
            background_num_samples = audio.get_num_samples(background_path)

            if background_num_samples > missing_num_samples:
                sample_offset = random.randint(
                    0, background_num_samples - missing_num_samples
                )
                num_samples = missing_num_samples
                background_samples = audio(
                    background_path, sample_offset=sample_offset, num_samples=num_samples
                )
                missing_num_samples = 0
            else:
                background_samples = audio(background_path)
                missing_num_samples -= background_num_samples

            pieces.append(background_samples)

        # the inner call to rms_normalize ensures concatenated pieces share the same RMS (1)
        # the outer call to rms_normalize ensures that the resulting background has an RMS of 1
        # (this simplifies "apply_transform" logic)
        return audio.rms_normalize(
            torch.cat([audio.rms_normalize(piece) for piece in pieces], dim=1)
        )

    def randomize_parameters(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        """

        :params samples: (batch_size, num_channels, num_samples)
        """

        batch_size, _, num_samples = samples.shape

        # (batch_size, num_samples) RMS-normalized background noise
        audio = self.audio if hasattr(self, "audio") else Audio(sample_rate, mono=True)
        self.transform_parameters["background"] = torch.stack(
            [self.random_background(audio, num_samples) for _ in range(batch_size)]
        )

        # (batch_size, ) SNRs
        if self.min_snr_in_db == self.max_snr_in_db:
            self.transform_parameters["snr_in_db"] = torch.full(
                size=(batch_size,),
                fill_value=self.min_snr_in_db,
                dtype=torch.float32,
                device=samples.device,
            )
        else:
            snr_distribution = torch.distributions.Uniform(
                low=torch.tensor(
                    self.min_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                high=torch.tensor(
                    self.max_snr_in_db, dtype=torch.float32, device=samples.device
                ),
                validate_args=True,
            )
            self.transform_parameters["snr_in_db"] = snr_distribution.sample(
                sample_shape=(batch_size,)
            )

    def apply_transform(
        self,
        samples: Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[Tensor] = None,
        target_rate: Optional[int] = None,
    ) -> ObjectDict:
        batch_size, num_channels, num_samples = samples.shape

        # (batch_size, num_samples)
        background = self.transform_parameters["background"].to(samples.device)

        # (batch_size, num_channels)
        background_rms = calculate_rms(samples) / (
            10 ** (self.transform_parameters["snr_in_db"].unsqueeze(dim=-1) / 20)
        )

        if self.add_noise_to_background:
            snr = torch.distributions.Uniform(
                low=torch.tensor(0.1, dtype=torch.float32, device=samples.device),
                high=torch.tensor(8.0, dtype=torch.float32, device=samples.device),
                validate_args=True,
            ).sample(sample_shape=(batch_size,))
            f_decay = torch.distributions.Uniform(
                low=torch.tensor(-2.0, dtype=torch.float32, device=samples.device),
                high=torch.tensor(2.0, dtype=torch.float32, device=samples.device),
                validate_args=True,
            ).sample(sample_shape=(batch_size,))

            # (batch_size, num_samples)
            noise = torch.stack(
                [
                _gen_noise(
                    f_decay[i],
                    num_samples,
                    sample_rate,
                    samples.device,
                )
                for i in range(batch_size)
                ]
            )

            # (batch_size, num_channels)
            noise_rms = calculate_rms(samples) / (
                10 ** (snr.unsqueeze(dim=-1) / 20)
            )
            background = background + noise_rms.unsqueeze(-1) * noise.view(batch_size, 1, num_samples).expand(-1, num_channels, -1)

        return ObjectDict(
            samples=samples
            + background_rms.unsqueeze(-1)
            * background.view(batch_size, 1, num_samples).expand(-1, num_channels, -1),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )

import time


def lp_degradation_(samples: Tensor):
    #we don't degrade the first sample for stability
    nb_samples_to_transform = samples.size(0) - 1
    samples_to_transform = samples[-nb_samples_to_transform:, ...]
    vinyl_background = AddBackgroundNoiseWithColoredNoise(background_paths=DATASET_PATH + "/augment/noise/vinyl", min_snr_in_db=3.0,
                                     max_snr_in_db=25.0, p=0.5, sample_rate=SAMPLE_RATE)

    short_noise = AddBackgroundNoise(background_paths=DATASET_PATH + "/augment/noise/other", min_snr_in_db=1.0,
                                     max_snr_in_db=20.0, p=0.33, sample_rate=SAMPLE_RATE)

    bandstop = BandStopFilter(p=0.2, min_bandwidth_fraction=0.05, max_bandwidth_fraction=1.0, sample_rate=SAMPLE_RATE)
    reverb = ApplyImpulseResponse(ir_paths=DATASET_PATH + "/augment/reverb", p=0.30, sample_rate=SAMPLE_RATE)
    clip = ClippingDistortionGPU(p=0.166666)
    lowpass = LowPassFilter(p=0.7, min_cutoff_freq=1000, max_cutoff_freq=16000, sample_rate=SAMPLE_RATE)
    colored_noise = AddColoredNoise(p=0.3, sample_rate=SAMPLE_RATE)
    white_noise = AddColoredNoise(p=0.9, min_f_decay=0, sample_rate=SAMPLE_RATE)

    backgrounds = [vinyl_background]
    eq_transform = [bandstop, reverb, lowpass]
    noises = [colored_noise, white_noise, short_noise]
    clip_distortion = [clip]
    random.shuffle(eq_transform)
    random.shuffle(noises)

    for transform in backgrounds + eq_transform + noises + clip_distortion:
        samples_to_transform = transform(samples_to_transform, sample_rate=SAMPLE_RATE)

    samples[-nb_samples_to_transform:, ...] = samples_to_transform
    return samples





class ClippingDistortionGPU(BaseWaveformTransform):
    supported_modes = {"per_batch", "per_example", "per_channel"}

    supports_multichannel = True
    requires_sample_rate = False
    supports_target = True
    requires_target = False

    def __init__(
            self,
            min_percentile_threshold: int = 0,
            max_percentile_threshold: int = 40,
            mode: str = "per_example",
            p: float = 0.5,
            p_mode: str = None,
    ):
        """
        :param min_percentile_threshold: int, A lower bound on the total percent of samples that
            will be clipped
        :param max_percentile_threshold: int, An upper bound on the total percent of samples that
            will be clipped
        :param mode:
        :param p:
        :param p_mode:
        """
        super().__init__(
            mode=mode,
            p=p,
            p_mode=p_mode,
        )
        assert min_percentile_threshold <= max_percentile_threshold, "min_percentile_threshold must not be greater than max_percentile_threshold"
        assert 0 <= min_percentile_threshold <= 100
        assert 0 <= max_percentile_threshold <= 100
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold
        self.randomize_parameters()

    def randomize_parameters(self,
                             samples: Tensor = None,
                             sample_rate: Optional[int] = None,
                             targets: Optional[Tensor] = None,
                             target_rate: Optional[int] = None,):
        self.transform_parameters["percentile_threshold"] = random.randint(
            self.min_percentile_threshold, self.max_percentile_threshold
        )

    def apply_transform(self, samples: Tensor = None,
                              sample_rate: Optional[int] = None,
                              targets: Optional[Tensor] = None,
                              target_rate: Optional[int] = None):
        lower_percentile_threshold = int(self.transform_parameters["percentile_threshold"] / 2)
        lower_threshold = torch.quantile(samples, lower_percentile_threshold / 100)
        upper_threshold = torch.quantile(samples, 1 - lower_percentile_threshold / 100)
        samples = torch.clamp(samples, lower_threshold, upper_threshold)
        return ObjectDict(
            samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


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


