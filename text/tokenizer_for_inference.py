import re
import torch
import pyopenjtalk

from .tokenizer import Tokenizer


class TokenizerForInfer:
    def __init__(self):
        self.max_a1 = 9
        self.min_a1 = -15
        self.max_f2 = 16
        self.tokenizer = Tokenizer()
        self.max_f2 = 16

        pyopenjtalk.extract_fullcontext('こんにちは')

    def __call__(self, s):
        labels = pyopenjtalk.extract_fullcontext(s)
        phoneme_list = list()
        a1_list = list()
        f2_list = list()
        for label in labels:
            if label.split("-")[1].split("+")[0] == "pau":
                phoneme_list += ["pau"]
                a1_list += ["xx"]
                f2_list += ["xx"]
                continue
            p = re.findall(r"\-(.*?)\+.*?\/A:([+-]?\d+).*?\/F:.*?_([+-]?\d+)", label)
            if len(p) == 1:
                phoneme, a1, f2 = p[0]
                phoneme = phoneme.lower() if phoneme != 'N' else phoneme
                if int(a1) > self.max_a1:
                    a1 = str(self.max_a1)
                if int(a1) < self.min_a1:
                    a1 = str(self.min_a1)
                if int(f2) > self.max_f2:
                    f2 = str(self.max_f2)
                phoneme_list += [phoneme]
                a1_list += [a1]
                f2_list += [f2]
        print(phoneme_list, a1_list, f2_list)

        phoneme, a1, f2 = self.tokenizer('_'.join(phoneme_list), '_'.join(a1_list), '_'.join(f2_list))
        phoneme, a1, f2 = phoneme.unsqueeze(0), a1.unsqueeze(0), f2.unsqueeze(0)
        length = torch.tensor([phoneme.size(1)], dtype=torch.long)
        return phoneme, a1, f2, length
