import torchaudio
import random
import torch
from torch import Tensor
import tempfile

def load_test_wav(directory='/Users/alex/dev/dataset/wav_chunks/', nb=8, mono=False):
    # load nb random wav files in directory and stack them into a batch
    wavs = []
    for file in random.sample(os.listdir(directory), nb):
        wav, sr = torchaudio.load(directory + file)
        if mono:
            wav = torch.mean(wav, dim=0, keepdim=True)
        wavs.append(wav)
    return torch.stack(wavs)


# play a sound from torchaudio tensor using default media player

def play_sound(sound: Tensor):
    torchaudio.io.play_audio(sound.T, 44100)


import os


def play_sound_vlc(sound: Tensor):
    # get temp dir
    directory = tempfile.gettempdir()
    file_path = directory + "/temp.wav"
    torchaudio.save(file_path, sound, 44100)
    os.system("/Applications/VLC.app/Contents/MacOS/VLC -I rc {}".format(file_path))


if __name__ == '__main__':
    wav = load_test_wav()
    play_sound_vlc(wav[0])

