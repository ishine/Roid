import re
import math
from argparse import ArgumentParser
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from transform import TacotronSTFT

ORIG_SR = None
NEW_SR = None


class PreProcessor:

    def __init__(self, config):
        self.wav_dir = Path(config.wav_dir)
        self.label_dir = Path(config.label_dir)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.to_mel = TacotronSTFT()

        global ORIG_SR, NEW_SR
        ORIG_SR = config.orig_sr
        NEW_SR = config.new_sr

    @staticmethod
    def get_time(label_path, sr=48000):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        b, e = lines[1], lines[-2]
        begin_time = int(int(b.split(' ')[0]) * 1e-7 * sr)
        end_time = int(int(e.split(' ')[1]) * 1e-7 * sr)
        return begin_time, end_time

    def load_wav(self, wav_path, label_path):
        wav, sr = sf.read(wav_path)
        b, e = self.get_time(label_path, sr=ORIG_SR)
        wav = wav[b:e]
        wav = librosa.resample(wav, ORIG_SR, NEW_SR)
        return wav

    @staticmethod
    def refine_duration(duration, y_length):
        diff = 0
        dur_fwd = list()
        diffs = list()
        for i in range(len(duration)):
            t_ = duration[i] + diff
            dur_fwd.append(round(t_))
            diff = t_ - dur_fwd[i]
            diffs.append(diff)
        dur_fwd[-1] += math.ceil(diff)
        dur_back = list()
        diff = 0
        for i in range(len(duration)-1, -1, -1):
            t_ = duration[i] + diff
            dur_back.append(round(t_))
            diff = t_ - dur_back[-1]
        dur_back[-1] += math.ceil(diff)
        duration = np.round(((np.array(dur_fwd) + np.array(list(reversed(dur_back)))) / 2)).astype(np.int32).tolist()
        sum_dur = sum(duration)
        if sum_dur < y_length:
            dur_diff = abs(sum_dur - y_length)
            for _ in range(dur_diff):
                idx = np.argmax(diffs)
                duration[idx] += np.sign(diffs[idx])
                diffs[idx] = 0
        elif sum_dur > y_length:
            dur_diff = abs(sum_dur - y_length)
            for _ in range(dur_diff):
                idx = np.argmin(diffs)
                duration[idx] += np.sign(diffs[idx])
                diffs[idx] = 0
        assert sum(duration) == y_length, f'{sum(duration)}, {y_length}'
        return duration

    def load_label(self, label_path, sr, y_length):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        phonemes, a1s, f2s, durations = list(), list(), list(), list()
        for line in lines:
            if line.split("-")[1].split("+")[0] == "pau":
                phonemes += ["pau"]
                a1s += ["xx"]
                f2s += ["xx"]

                b, e = line.split()[:2]
                b, e = int(b), int(e)
                duration = (e - b) * 1e-7 * sr
                duration = duration / 256
                durations += [duration]
                continue
            p = re.findall(r"\-(.*?)\+.*?\/A:([+-]?\d+).*?\/F:.*?_([+-]?\d+)", line)
            if len(p) == 1:
                b, e = line.split()[:2]
                b, e = int(b), int(e)
                duration = (e - b) * 1e-7 * sr
                duration = duration / 256
                durations += [duration]

                phoneme, a1, f2 = p[0]
                phonemes += [phoneme]
                a1s += [a1]
                f2s += [f2]
        assert len(phonemes) == len(a1s) and len(phonemes) == len(f2s)
        durations = self.refine_duration(durations, y_length)
        return phonemes, a1s, f2s, durations

    @staticmethod
    def extract_feats(wav):
        f0, sp, ap = pw.wav2world(wav, NEW_SR, 1024, 256 / NEW_SR * 1000)
        return f0, sp, ap

    def process_speaker(self, wav_dir_path, label_dir_path):
        wav_paths = list(sorted(list(wav_dir_path.glob('*.wav'))))
        label_paths = list(sorted(list(label_dir_path.glob('*.lab'))))

        wavs = list()
        mels = list()
        labels = list()
        lengths = list()
        pitches = list()
        energies = list()
        durations = list()

        for i in tqdm(range(len(wav_paths))):
            wav = self.load_wav(wav_paths[i], label_paths[i])
            pitch, *_ = self.extract_feats(wav)
            mel, energy = self.to_mel(torch.FloatTensor(wav)[None, :])
            *label, duration = self.load_label(label_paths[i], NEW_SR, mel.size(-1))

            assert sum(duration) == mel.size(-1), f'{sum(duration)}, {mel.size(-1)}'

            pitch = np.array(pitch).astype(np.float32)
            energy = np.array(energy).astype(np.float32)

            pitch[pitch != 0] = np.log(pitch[pitch != 0])
            energy[energy != 0] = np.log(energy[energy != 0])

            assert pitch.shape[0] == mel.size(-1)

            wavs.append(wav)
            mels.append(mel)
            labels.append(label)
            lengths.append(mel.size(-1))
            pitches.append(pitch)
            energies.append(energy)
            durations.append(duration)

        return wavs, mels, labels, pitches, energies, lengths, durations

    def preprocess(self):
        print('Start wav')
        wavs, mels, labels, pitches, energies, lengths, durations = self.process_speaker(self.wav_dir, self.label_dir)

        print('Save file')
        for i in tqdm(range(len(mels))):
            torch.save([
                torch.FloatTensor(wavs[i])[None, :],
                mels[i].squeeze().transpose(0, 1),
                labels[i],
                lengths[i],
                torch.FloatTensor(pitches[i]).view(-1, 1),
                torch.FloatTensor(energies[i]).view(-1, 1),
                torch.LongTensor(durations[i]).view(-1, 1),
            ], self.output_dir / f'data_{i+1:04d}.pt')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/preprocess.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    PreProcessor(config).preprocess()
