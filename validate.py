import os
import torch
import torchaudio
import matplotlib.pyplot as plt

from tqdm import tqdm
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf

from models import TTSModel
from hifi_gan import load_hifi_gan
from text import Tokenizer
from transform import TacotronSTFT

SR = 24000


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--hifi_gan', type=str, required=True)
    parser.add_argument('--data_file_path', type=str, default='./filelists/data.txt')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args = parser.parse_args()

    config = OmegaConf.load(f'{args.model_dir}/config.yaml')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(f'{args.model_dir}/latest.ckpt', map_location=device)
    model = TTSModel(config.model)
    model.load_state_dict(checkpoint['model'])
    print(f'Loaded {checkpoint["iteration"]} Iteration Model')
    hifi_gan = load_hifi_gan(args.hifi_gan)
    model, hifi_gan = model.eval().to(device), hifi_gan.eval().to(device)

    tokenizer = Tokenizer()
    to_mel = TacotronSTFT()

    def load_audio(wav_path):
        wav = torchaudio.load(wav_path)[0]
        mel = to_mel(wav)
        return wav, mel

    def infer(label):
        phoneme, a1, f2 = tokenizer(*label)
        phoneme, a1, f2 = phoneme.unsqueeze(0).to(device), a1.unsqueeze(0).to(device), f2.unsqueeze(0).to(device)
        length = torch.LongTensor([phoneme.size(-1)]).to(device)
        with torch.no_grad():
            mel = model.infer(phoneme, a1, f2, length)
            wav = hifi_gan(mel)
            mel, wav = mel.cpu(), wav.squeeze(1).cpu()
        return mel, wav

    def save_wav(wav, path):
        torchaudio.save(
            str(path),
            wav,
            24000,
            encoding='PCM_S',
            bits_per_sample=16
        )

    def save_mel_two(gen, gt, path):
        plt.figure(figsize=(10, 7))
        plt.subplot(211)
        plt.gca().title.set_text('GEN')
        plt.imshow(gen, aspect='auto', origin='lower')
        plt.subplot(212)
        plt.gca().title.set_text('GT')
        plt.imshow(gt, aspect='auto', origin='lower')
        plt.savefig(path)
        plt.close()

    with open(args.data_file_path, 'r') as f:
        lines = f.readlines()
    lines = lines[:100]

    for line in tqdm(lines, total=len(lines)):
        wav_path, *label = line.strip().split('|')
        wav, mel = load_audio(wav_path)
        mel_gen, wav_gen = infer(label)

        d = output_dir / os.path.splitext(os.path.basename(wav_path))[0]
        d.mkdir(exist_ok=True)

        save_wav(wav, d / 'gt.wav')
        save_wav(wav_gen, d / 'gen.wav')

        save_mel_two(mel_gen.squeeze(), mel.squeeze(), d / 'comp.png')


if __name__ == '__main__':
    main()
