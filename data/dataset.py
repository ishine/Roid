import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from text import Tokenizer


class VCDataset(Dataset):
    def __init__(self, fns):
        self.fns = fns
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        (
            _,
            mel,
            label,
            length,
            pitch,
            energy,
            duration
        ) = torch.load(self.fns[idx])
        phoneme, a1, f2 = self.tokenizer(*label)
        return mel, phoneme, a1, f2, pitch, energy, duration, length


def collate_fn(batch):
    (
        mel,
        phoneme,
        a1,
        f2,
        pitch,
        energy,
        duration,
        y_length
    ) = tuple(zip(*batch))

    mel = pad_sequence(mel, batch_first=True).transpose(-1, -2)
    phoneme = pad_sequence(phoneme, batch_first=True)
    a1 = pad_sequence(a1, batch_first=True)
    f2 = pad_sequence(f2, batch_first=True)
    x_length = torch.LongTensor([len(x) for x in phoneme])

    pitch = pad_sequence(pitch, batch_first=True).transpose(-1, -2)
    energy = pad_sequence(energy, batch_first=True).transpose(-1, -2)
    duration = pad_sequence(duration, batch_first=True).transpose(-1, -2)
    y_length = torch.LongTensor(y_length)

    return (
        mel,
        phoneme,
        a1,
        f2,
        pitch,
        energy,
        duration,
        x_length,
        y_length
    )
