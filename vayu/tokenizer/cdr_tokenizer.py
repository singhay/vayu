import json
import logging
import os
import unicodedata
import re
from typing import List, Union

from vayu.tokenizer.stopwords import stopword_normalization_map

logger = logging.getLogger(__name__)

BOS, EOS, PAD, MASK, UNK = "<s>", "</s>", "<pad>", "<mask>", "<unk>"


class CDRTokenizer:
    """WIP Tokenizer that assumes a vocab / whitelist file to be available"""
    VOCAB_FILE = "vocab.json"

    def __init__(self, path, lowercase=True, stem=False, normalize=True, normalization_map=List[dict]):
        files = set(os.listdir(path))
        if self.VOCAB_FILE in files:
            self.vocab = json.load(open(os.path.join(path, self.VOCAB_FILE)))
        else:
            # TODO: implement build_vocab() in case vocab is not provided
            raise FileNotFoundError(f"No {self.VOCAB_FILE} file found")
        self.is_stem = stem
        self.is_lowercase = lowercase
        self.is_normalize = normalize
        self.normalization_map = normalization_map

        self.pad_token, self.pad_token_id = self.vocab.get(PAD, -1), PAD
        self.bos_token, self.bos_token_id = self.vocab.get(BOS, -1), BOS
        self.eos_token, self.eos_token_id = self.vocab.get(EOS, -1), EOS
        self.mask_token, self.mask_token_id = self.vocab.get(MASK, -1), MASK
        self.unk_token, self.unk_token_id = self.vocab.get(UNK, -1), UNK

    def _tokenize(self, sentence: str) -> List[str]:
        return list(map(lambda x: x if x in self.vocab else self.unk_token,
                        sentence.split(" ")))

    def vocab_size(self):
        return len(self.vocab)

    def encode_batch(self, document: Union[List[List[str]], List[str]]) -> List[List[str]]:
        if all(isinstance(i, list) for i in document):
            return document
        else:
            return list(map(self.encode, document))

    def encode(self, sentence: str) -> List[str]:
        # Encode already tokenized sequence; backward compatible with docs from NTA pipeline-runner
        if isinstance(sentence, list):
            return self._tokens_to_ids(sentence)
        else:
            sentence = self.remove_non_ascii(sentence)

            if self.is_lowercase:
                # TODO: if len(word) >= 3:
                sentence = sentence.lower().strip().rstrip("\n")
            if self.is_normalize:
                sentence = self.normalize(sentence)
            if self.is_stem:
                sentence = self.stem(sentence)

            sentence = self._tokenize(sentence)
            sentence = map(self.normalize_stopwords, sentence)
            return [self.bos_token_id] + self._tokens_to_ids(sentence) + [self.eos_token_id]

    @staticmethod
    def remove_non_ascii(text):
        """Remove non-ASCII characters from list of tokenized words"""
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

    def normalize(self, text: str):
        """Run regex that replace found patterns with standardized tag"""
        for regex, tag in self.normalization_map:
            self.vocab[tag] = len(self.vocab)
            text = re.sub(repr(regex), tag, text)
        return text

    @staticmethod
    def normalize_stopwords(word: str) -> str:
        """Return corresponding stop word tag if found else return original word"""
        for word_set, tag in stopword_normalization_map:
            if word in word_set:
                return tag
        return word

    def stem(self, word: str):
        """TODO: return stem + removed_characters"""
        pass
