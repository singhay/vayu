"""
Train tokenizer
===============

Script used to train the tokenizer for RoBERTa

Improvements
------------
* TODO: Extend script to train BERT (wordpiece) and ALBERT (sentencepiece)

"""

import argparse
import os
from glob import glob

from tokenizers import ByteLevelBPETokenizer

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True,
                                  continuing_subword_prefix="#",
                                  end_of_word_suffix="$",
                                  trim_offsets=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The input training data directory containing single or multiple text files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the trained tokenizer files will be saved.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30522,
        help="The output directory where the trained tokenizer files will be saved.",
    )
    parser.add_argument(
        "--min_frequency",
        type=str,
        default=10,
        help="The output directory where the trained tokenizer files will be saved.",
    )
    args = parser.parse_args()

    # Customize training
    tokenizer.train(files=glob(os.path.join(args.data_dir, "*.txt")),
                    vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
            # Order of following special tokens is important
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

    # Save files to disk
    tokenizer.save(directory=args.output_dir)
