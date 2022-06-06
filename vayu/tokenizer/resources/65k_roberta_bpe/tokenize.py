from tokenizers import ByteLevelBPETokenizer

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()


if __name__ == "__main__":
    # Customize training
    tokenizer.train(files=["/home/data/BPE/txt_data/72148_tokens.txt",
                           "/home/data/BPE/txt_data/70551_tokens.txt",
                           "/home/data/BPE/txt_data/70553_tokens.txt",
                           "/home/data/BPE/txt_data/78452_tokens.txt",
                           "/home/data/BPE/txt_data/74177_tokens.txt",
                           "/home/data/BPE/txt_data/71260_tokens.txt",
                           "/home/data/BPE/txt_data/71250_tokens.txt"],
                    vocab_size=65536,
                    min_frequency=10,
                    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    # Save files to disk
    tokenizer.save("./65k_roberta")
