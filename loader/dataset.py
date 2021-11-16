import torch
import torch.nn as nn
from torchvision import transforms
import json
import random
from PIL import Image

from .auto_augment import ImageNetPolicy


class CaptioningDataset(nn.Dataset):
    def __init__(self, json_annot_path, inp_img_size, max_seq, vocab, phase):
        super().__init__()
        self.json_ann = json_annot_path
        self.phase = phase
        self.max_seq = max_seq
        self.vocab = vocab
        
        with open(json_annot_path, "r") as f:
            self.data = json.load(f)[phase]
        
        if phase != "test" and phase != "val": 
            self.transforms = transforms.Compose([
                transforms.Resize(inp_img_size),
                ImageNetPolicy(), # Auto Augmentation for ImageNet/COCO
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        else: # Train
            self.transforms = transforms.Compose([
                transforms.Resize(inp_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        try:
            img_path = self.data[index]["filepath"]
            img = Image.open(img_path)
            img = self.transforms(img)

            captions_number = self.data[index]["captions_number"]
            token = self.data[index]["sentences"][random.randint(0, captions_number-1)]
            if len(token) > self.max_seq:
                return self.__getitem__(index + 1)
            caption_tensor = torch.Tensor([self.vocab(self.vocab.start_word)] + \
                                            [self.vocab(w) for w in token] + \
                                            [self.vocab(self.vocab.end_word)] + \
                                            [self.vocab(self.vocab.pad_word)] * (self.max_seq-len(token))).long()
        except Exception as exception:
            print('Exception: ', exception)
            print('Error index: ', index)
            return self.__getitem__(index + 1)

        return img, caption_tensor






        
