
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from pathlib import Path
import subprocess

from dora.log import fatal
import torch
import torch as th
import torchaudio as ta

import piano_demucs
from demucs.apply import apply_model
from demucs.audio import AudioFile, convert_audio, save_audio
import piano_demucs


def get_model():
    model = piano_demucs.get_model()
    checkpoint = Path("best.th")
    package = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(package["state"])
    model.eval()
    return model


def load_track(track, audio_channels, samplerate):
    errors = {}
    wav = None

    try:
        wav = AudioFile(track).read(
            streams=0,
            samplerate=samplerate,
            channels=audio_channels)
    except FileNotFoundError:
        errors['ffmpeg'] = 'FFmpeg is not installed.'
    except subprocess.CalledProcessError:
        errors['ffmpeg'] = 'FFmpeg could not read the file.'

    if wav is None:
        try:
            wav, sr = ta.load(str(track))
        except RuntimeError as err:
            errors['torchaudio'] = err.args[0]
        else:
            wav = convert_audio(wav, sr, samplerate, audio_channels)

    if wav is None:
        print(f"Could not load file {track}. "
              "Maybe it is not a supported file format? ")
        for backend, error in errors.items():
            print(f"When trying to load using {backend}, got the following error: {error}")
        sys.exit(1)
    return wav




def main(track="/Users/alex/dev/dataset/piano_restauration/test/hungho.flac", nb_steps=1):
    track = Path(track)
    out_dir = Path("/Users/alex/dev/dataset/piano_restauration/test/denoised")

    out_file = track.name.rsplit(".", 1)[0] + "_denoised_" + str(nb_steps) + "." + track.name.rsplit(".", 1)[-1]
    file_out: Path = out_dir / out_file
    model = get_model()

    model = model.to('mps')
    model.eval()


    print(f"Denoised tracks will be stored in {file_out.resolve()}")
    if not track.exists():
        print(f"File {track} does not exist.", file=sys.stderr)
        return
    print(f"Denoising track {track}")
    wav = load_track(track, model.audio_channels, model.samplerate)
    #convert to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(0,keepdim=True)
    wav = wav.to('mps')
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()
    sources = wav[None]
    for i in range(nb_steps):
        sources = apply_model(model, sources, device='mps', shifts=0,
                              split=True, overlap=0.25, progress=True,
                              num_workers=6, segment=10)[0]
    sources.squeeze_(0)
    sources *= ref.std()
    sources += ref.mean()
    kwargs = {
         'samplerate': model.samplerate,
         'bitrate': 320,
          'bits_per_sample':  16,
    }

    sources = sources.to('cpu')
    save_audio(sources, file_out, **kwargs)


if __name__ == "__main__":
    main(nb_steps=3)
