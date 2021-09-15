import torch


class Tokenizer:
    def __init__(self, dictionary_path='./filelists/word_index.txt', state_size=1):
        self.a1_coef = 15
        self.state_size = state_size
        self.dictionary = self.load_dictionary(dictionary_path)
        self.accent_dict = self.build_accent_dict()

    def __call__(self, phonemes, a1s, f2s):
        phonemes = [[self.dictionary[s]+len(self.dictionary)*i for i in range(self.state_size)] for s in phonemes]
        phonemes = sum(phonemes, [])

        a1s = [a1s[i + 1] if i == 0 and a1 == 'xx' else a1s[i - 1] if a1 == 'xx' else a1 for i, a1 in enumerate(a1s)]
        a1s = [int(a1) / self.a1_coef for a1 in a1s]
        a1s = sum([[a1 for _ in range(self.state_size)] for a1 in a1s], [])

        f2s = [self.accent_dict[f2] for f2 in f2s]
        f2s = sum([[f2 for _ in range(self.state_size)] for f2 in f2s], [])

        phonemes = torch.LongTensor(phonemes)
        a1s = torch.FloatTensor(a1s)
        f2s = torch.LongTensor(f2s)
        return phonemes, a1s, f2s

    @staticmethod
    def load_dictionary(path):
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        dictionary = dict()
        for i, w in enumerate([w.strip() for w in lines]):
            dictionary[w] = i
        return dictionary

    @staticmethod
    def build_accent_dict():
        d = {str(k): i for i, k in enumerate(range(0, 16+1), start=1)}
        d['xx'] = len(d)+1
        return d

    def __len__(self):
        return len(self.dictionary)
