"""
Tokenization mixins
===================

Set of mixins that enable tokenization flat and nested (chunked) documents

"""

from typing import List

import torch


class FlatDatasetTokenizationMixin:
    """Mixin class for tokenizing nested documents into flat documents

    :param int max_length: Maximum length of the flat documents returned (max_chunks * max_chunk_size)
    """

    def __init__(self, max_length):
        self.max_length = max_length

    def tokenize(self, doc: List[List[str]]) -> (List[int], int):
        """Chunk, tokenize and pad a document of sequences

        :param doc: list[list[str]] cdr document that is sentenized -> tokenized using legacy pipeline
        :return: tuple (padded sequences, length before padding)
        :rtype: (List[int], int)
        """
        if not doc:  # handle empty docs
            raise ValueError("Empty document passed, please clean dataset or set data.is_lazy=false")

        flat_doc = ' '.join([token for sent in doc for token in sent])
        inputs = self.tokenizer.encode(flat_doc, return_tensors='pt')
        return inputs[0][:self.max_length], len(inputs[0][:self.max_length])


class ChunkedDatasetTokenizationMixin:
    """Mixin class for tokenizing nested documents into nested documents using custom tokenizer

    :param int max_chunks: int Maximum number of chunks allowed when tokenizing
    :param int max_chunk_size: int Maximum size of chunk to keep when tokenizing
    """

    def __init__(self, max_chunks: int, max_chunk_size: int):
        self.max_chunks = max_chunks
        self.max_chunk_size = max_chunk_size

    def tokenize(self, doc: List[List[str]]) -> (List[List[int]], int):
        """Chunk, tokenize and pad a document of sequences

        :param doc: list[list[str]] cdr document that is sentenized -> tokenized using legacy pipeline
        :return: tuple (padded chunks (sequences of sentences), length before padding)
        :rtype: (List[List[int]], int)
        """
        chunks = []

        if not doc:  # handle empty docs
            raise ValueError("Empty document passed, please clean dataset or set data.is_lazy=false")

        tokenized_doc = self.tokenizer.batch_encode_plus([' '.join(sent) for sent in doc],
                                                         return_token_type_ids=False,
                                                         return_attention_mask=False, truncation=True,
                                                         max_length=self.max_chunk_size)['input_ids']

        chunk, chunk_len = [], 0
        for sent in tokenized_doc:
            curr_sent_len = len(sent)
            if chunk_len + curr_sent_len < self.max_chunk_size:
                chunk_len += curr_sent_len
                chunk += sent
            else:
                # Pad chunk
                chunks.append(chunk + [self.tokenizer.pad_token_id] * (self.max_chunk_size - len(chunk)))
                chunk, chunk_len = sent, curr_sent_len

        # Handle case when document does not even have self.max_chunk_size chunks
        if not chunks:
            chunks.append(chunk + [self.tokenizer.pad_token_id] * (self.max_chunk_size - len(chunk)))

        return torch.LongTensor(chunks[:self.max_chunks]), len(chunks[:self.max_chunks])


class ChunkedFirstDatasetTokenizationMixin:
    """

    :param int max_chunks: int Maximum number of chunks allowed when tokenizing
    :param int max_chunk_size: int Maximum size of chunk to keep when tokenizing
    """
    def __init__(self, max_chunks: int, max_chunk_size: int):
        self.max_chunks = max_chunks
        self.max_chunk_size = max_chunk_size

    def tokenize(self, doc: List[List[str]]):
        """Chunk, tokenize and pad a document of sequences

        :param doc: list[list[str]] cdr document that is sentenized -> tokenized using legacy pipeline
        :return: tuple (padded sequences, length before padding)

        TODO: Generate overlapping chunks to handle context fragmentation
        """
        flat_doc = [token for sent in doc for token in sent]
        tokenized = self.tokenizer.tokenize(' '.join(flat_doc))
        chunks = [' '.join(tokenized[i:i + self.max_chunk_size]) for i in range(0, len(tokenized), self.max_chunk_size)]
        inputs = self.tokenizer.batch_encode_plus(chunks[:self.max_chunks], return_tensors='pt',
                                                  truncation=True,
                                                  max_length=self.max_chunk_size, pad_to_max_length=True)
        return inputs['input_ids'], len(chunks[:self.max_chunks])

