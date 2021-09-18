import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import torchaudio

from text import Tokenizer
from transform import TacotronSTFT


class TTSDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = Tokenizer()
        self.to_mel = TacotronSTFT()

    def __len__(self):
        return len(self.data)

    def load_mel(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        wav = wav + torch.rand_like(wav) / 32768.0
        mel = self.to_mel(wav).squeeze().transpose(-1, -2)
        return mel

    def __getitem__(self, idx):
        wav_path, *label = self.data[idx].strip().split('|')
        phoneme, a1, f2 = self.tokenizer(*label)
        mel = self.load_mel(wav_path)
        return phoneme, a1, f2, mel


def collate_fn(batch):
    phoneme, a1, f2, mel = tuple(zip(*batch))

    x_length = torch.LongTensor([len(x) for x in phoneme])
    phoneme = pad_sequence(phoneme, batch_first=True)
    a1 = pad_sequence(a1, batch_first=True)
    f2 = pad_sequence(f2, batch_first=True)

    y_length = torch.LongTensor([len(x) for x in mel])
    mel = pad_sequence(mel, batch_first=True).transpose(-1, -2)

    return (
        phoneme,
        a1,
        f2,
        x_length,
        mel,
        y_length
    )
