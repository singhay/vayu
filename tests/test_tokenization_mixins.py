import pytest
import torch
from transformers import RobertaTokenizerFast

from vayu.tokenizer import FlatDatasetTokenizationMixin, ChunkedDatasetTokenizationMixin


def test_flat_tokenization_mixin_empty_document():
    doc = []
    flat_data_tokenizer = FlatDatasetTokenizationMixin(max_length=50)

    with pytest.raises(ValueError,
                       match="Empty document passed, please clean dataset or set data.is_lazy=false"):
        _ = flat_data_tokenizer.tokenize(doc)


def test_flat_tokenization_mixin_small_document():
    doc = [['<', 'pagebreak', '>']]
    flat_data_tokenizer = FlatDatasetTokenizationMixin(max_length=50)
    bpe_tokenizer = RobertaTokenizerFast.from_pretrained("./vayu/tokenizer/resources/65k_roberta_bpe")
    flat_data_tokenizer.tokenizer = bpe_tokenizer

    expected_doc = bpe_tokenizer.encode(' '.join(doc[0]))
    tokenized_doc, tokenized_len = flat_data_tokenizer.tokenize(doc)
    assert tokenized_doc.tolist() == expected_doc
    assert tokenized_len == len(expected_doc)


def test_flat_tokenization_mixin_max_length_truncate():
    doc = [['<', 'pagebreak', '>'], ['']]*50
    max_length = 100
    flat_data_tokenizer = FlatDatasetTokenizationMixin(max_length=max_length)
    bpe_tokenizer = RobertaTokenizerFast.from_pretrained("./vayu/tokenizer/resources/65k_roberta_bpe")
    flat_data_tokenizer.tokenizer = bpe_tokenizer

    tokenized_doc, tokenized_len = flat_data_tokenizer.tokenize(doc)
    assert tokenized_len == max_length


def test_flat_tokenization_mixin_within_max_length():
    doc = [['<', 'pagebreak', '>'], ['']]*50
    max_length = 300
    flat_data_tokenizer = FlatDatasetTokenizationMixin(max_length=max_length)
    bpe_tokenizer = RobertaTokenizerFast.from_pretrained("./vayu/tokenizer/resources/65k_roberta_bpe")
    flat_data_tokenizer.tokenizer = bpe_tokenizer

    tokenized_doc, tokenized_len = flat_data_tokenizer.tokenize(doc)
    assert tokenized_len < max_length


def test_chunked_tokenization_mixin_small_document():
    doc = [['<', 'pagebreak', '>']]
    max_chunks, max_chunk_size = 10, 32
    chunked_data_tokenizer = ChunkedDatasetTokenizationMixin(max_chunks=max_chunks,
                                                             max_chunk_size=max_chunk_size)
    bpe_tokenizer = RobertaTokenizerFast.from_pretrained("./vayu/tokenizer/resources/65k_roberta_bpe")
    chunked_data_tokenizer.tokenizer = bpe_tokenizer

    expected_doc = bpe_tokenizer.encode(' '.join(doc[0]))
    expected_doc += [bpe_tokenizer.pad_token_id] * (max_chunk_size - len(expected_doc))
    expected_doc = [expected_doc]
    tokenized_doc, tokenized_len = chunked_data_tokenizer.tokenize(doc)
    assert tokenized_doc.tolist() == expected_doc
    assert tokenized_len == len(expected_doc)


def test_chunked_tokenization_mixin_max_length_truncate():
    doc = [['<', 'pagebreak', '>'], ['']]*50
    max_chunks, max_chunk_size = 10, 32
    chunked_data_tokenizer = ChunkedDatasetTokenizationMixin(max_chunks=max_chunks,
                                                             max_chunk_size=max_chunk_size)
    bpe_tokenizer = RobertaTokenizerFast.from_pretrained("./vayu/tokenizer/resources/65k_roberta_bpe")
    chunked_data_tokenizer.tokenizer = bpe_tokenizer

    tokenized_doc, tokenized_len = chunked_data_tokenizer.tokenize(doc)
    assert tokenized_doc.size() == torch.randint(0, 3, (10, 32)).size()
    assert tokenized_len == max_chunks


def test_chunked_tokenization_mixin_within_max_length():
    doc = [['<', 'pagebreak', '>'], ['']]*7
    max_chunks, max_chunk_size = 20, 16
    expected_pad_len = 5
    chunked_data_tokenizer = ChunkedDatasetTokenizationMixin(max_chunks=max_chunks,
                                                             max_chunk_size=max_chunk_size)
    bpe_tokenizer = RobertaTokenizerFast.from_pretrained("./vayu/tokenizer/resources/65k_roberta_bpe")
    chunked_data_tokenizer.tokenizer = bpe_tokenizer

    tokenized_doc, tokenized_len = chunked_data_tokenizer.tokenize(doc)
    print(tokenized_doc)
    assert tokenized_doc.size() == torch.randint(0, 3, (4, max_chunk_size)).size()
    assert torch.equal(tokenized_doc[1, max_chunk_size-expected_pad_len:], torch.tensor([1]*expected_pad_len))  # make sure padding is correct
    assert tokenized_len == 4
    assert tokenized_len < max_chunks
