import re
from argparse import ArgumentParser
from pathlib import Path

import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm

ORIG_SR = None
NEW_SR = None


class PreProcessor:

    def __init__(self, config):
        self.wav_dir = Path(config.wav_dir)
        self.label_dir = Path(config.label_dir)

        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        global ORIG_SR, NEW_SR
        ORIG_SR = config.orig_sr
        NEW_SR = config.new_sr

        self.resample = torchaudio.transforms.Resample(ORIG_SR, NEW_SR)

    @staticmethod
    def get_time(label_path, sr=48000):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        b, e = lines[1], lines[-2]
        begin_time = int(int(b.split(' ')[0]) * 1e-7 * sr)
        end_time = int(int(e.split(' ')[1]) * 1e-7 * sr)
        return begin_time, end_time

    def load_wav(self, wav_path, label_path):
        wav, sr = torchaudio.load(wav_path)
        b, e = self.get_time(label_path, sr=ORIG_SR)
        wav = wav[:, b:e]
        wav = self.resample(wav)
        return wav

    def load_label(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        phonemes, a1s, f2s, durations = list(), list(), list(), list()
        for line in lines:
            if line.split("-")[1].split("+")[0] == "pau":
                phonemes += ["pau"]
                a1s += ["xx"]
                f2s += ["xx"]
                continue
            p = re.findall(r"\-(.*?)\+.*?\/A:([+-]?\d+).*?\/F:.*?_([+-]?\d+)", line)
            if len(p) == 1:
                phoneme, a1, f2 = p[0]
                phonemes += [phoneme]
                a1s += [a1]
                f2s += [f2]
        assert len(phonemes) == len(a1s) and len(phonemes) == len(f2s)
        return phonemes, a1s, f2s

    def process_speaker(self, wav_dir_path, label_dir_path):
        wav_paths = list(sorted(list(wav_dir_path.glob('*.wav'))))
        label_paths = list(sorted(list(label_dir_path.glob('*.lab'))))

        labels = list()
        for i in tqdm(range(len(wav_paths))):
            wav = self.load_wav(wav_paths[i], label_paths[i])
            phoneme, a1, f2 = self.load_label(label_paths[i])
            labels.append(f'DATA/{wav_paths[i].name}|{"_".join(phoneme)}|{"_".join(a1)}|{"_".join(f2)}\n')
            # torchaudio.save(
            #     str(self.output_dir / wav_paths[i].name),
            #     wav,
            #     NEW_SR,
            #     encoding='PCM_S',
            #     bits_per_sample=16
            # )
        with open('./filelists/data.txt', 'w', encoding='utf-8') as f:
            f.writelines(labels)

    def preprocess(self):
        self.process_speaker(self.wav_dir, self.label_dir)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/preprocess.yaml')
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    PreProcessor(config).preprocess()
