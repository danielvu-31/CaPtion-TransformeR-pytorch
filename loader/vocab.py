from tqdm import tqdm
import nltk
import pickle
import json
from collections import Counter
from spacy.lang.en import English


class Vocab():
    '''
    Define a vocabulary file for tokenization
    '''

    punctuations = ["''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", \
                    ".", "?", "!", ",", ":", "-", "--", "...", ";"]
    
    def __init__(self,
                min_threshold, 
                annotation_file_path,
                splits,
                remove_punctuations=True,
                start_word="<start>",
                end_word="<end>",
                unk_word="<unk>",
                pad_word="<pad>",
                vocab_from_file=False,
                vocab_file="./vocab.pkl",
                tokenizer="nltk"):
        self.min_threshold = min_threshold
        self.annotation_file_path = annotation_file_path
        self.tokenizer = tokenizer
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.vocab_from_file = vocab_from_file
        self.vocab_file = vocab_file
        self.splits = splits
        self.remove_punctuations = remove_punctuations

        with open(self.annotation_file_path, 'r') as f:
            self.full_data = json.load(f)
        
        if not vocab_from_file:
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers
        (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def build_vocab(self):
        self.init_vocab()
        self.add_word(self.pad_word)
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.get_full_vocab()

    def add_word(self, word):
        if word not in self.word2idx.keys():
            self.word2idx[word] = self.idx
            self.idx2word[self.idx2word] = word
            self.idx += 1

    def get_full_vocab(self):
        counter = Counter()

        for s in self.splits:
            data = self.full_data[s]
            image_ids = tqdm(data, desc=f'[Tokenizing {s}]:')
            for _, item in enumerate(image_ids):
                for token in item["sentences"]:
                    if self.tokenizer == "nltk":
                        tokens = nltk.tokenize.word_tokenize(token["raw"].lower().strip())
                    else:
                        nlp = English()
                        tokenizer = nlp.tokenizer
                        tokens = tokenizer(token["raw"].lower().strip())
                    if self.remove_punctuations:
                        tokens = [token for token in tokens if token not in self.punctuations]
                    counter.update(tokens)

        words = [word for word, count in counter.items() if count >= self.min_threshold]
        for _, word in enumerate(words):
            self.add_word(word)
        
    def __len__(self):
        return len(self.word2idx)
    
    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]


