import os
import json
import random
from pathlib import Path
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import sentencepiece as spm


# EXTRACT COMPOUNDS
def extract_compound_data(root_dir):
    data = []
    for file in Path(root_dir).rglob("*.conllu"):
        with open(file, encoding="utf-8") as f:
            lines = f.readlines()
        text = None
        compounds = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("# text ="):
                text = line.replace("# text =", "").strip()
                compounds = []
            elif line and not line.startswith("#"):
                parts = line.split("\t")
                if "-" in parts[0]:
                    start, end = map(int, parts[0].split("-"))
                    subwords = []
                    for j in range(i + 1, len(lines)):
                        nxt = lines[j].strip()
                        if not nxt or nxt.startswith("#"):
                            break
                        subparts = nxt.split("\t")
                        if "-" in subparts[0]:
                            break
                        idx = int(subparts[0])
                        if start <= idx <= end:
                            subwords.append(subparts[1])
                        if idx > end:
                            break
                    if len(subwords) >= 2:
                        compounds.append(subwords)
            elif line == "" and text:
                if compounds:
                    data.append((text, compounds))
                text = None
                compounds = []
            i += 1
    return data


# TRANSLITERATION
def convert_text(text, scheme):
    return transliterate(text, sanscript.IAST, scheme)

def convert_dataset(data, scheme):
    new_data = []
    for text, compounds in data:
        new_text = convert_text(text, scheme)
        new_compounds = [[convert_text(p, scheme) for p in parts] for parts in compounds]
        new_data.append((new_text, new_compounds))
    return new_data


# SPLIT
def split_data(data, test_ratio=0.2, seed=42):
    data = data.copy()
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    return train_data, test_data

import random
def sample_test_from_train(train_data, test_ratio=0.2, seed=42):
    random.seed(seed)
    train_copy = train_data.copy()
    random.shuffle(train_copy)
    split_idx = int(len(train_copy) * test_ratio)
    test_subset = train_copy[:split_idx]
    return test_subset


# TOKENIZATION HELPERS
def tokenize_hf(tokenizer, text):
    return tokenizer.encode(text).tokens

def tokenize_sp(sp, text):
    tokens = sp.encode(text, out_type=str)
    return [t.replace("▁", "") for t in tokens if t.strip()]


# COMPOUND CHECK
def check_compounds(tokenizer, data, mode="hf"):
    total = 0
    correct = 0
    for text, compounds in data:
        for parts in compounds:
            total += 1
            # reconstruct compound word
            compound_word = "".join(parts)
            # tokenize ONLY the compound
            if mode == "sp":
                tokens = tokenize_sp(tokenizer, compound_word)
            else:
                tokens = tokenize_hf(tokenizer, compound_word)
            # exact match check
            if tokens == parts:
                correct += 1
    return correct, total

# FERTILITY
def compute_fertility(tokenizer, data, mode="hf"):
    total_tokens = 0
    total_words = 0
    for text, _ in data:
        words = text.split()
        total_words += len(words)
        tokens = tokenize_sp(tokenizer, text) if mode == "sp" else tokenize_hf(tokenizer, text)
        total_tokens += len(tokens)
    return total_tokens / total_words if total_words > 0 else 0

# EVALUATION
def evaluate_all(name, data, bpe_tok, sp_tok):
    bpe_correct, total = check_compounds(bpe_tok, data, mode="hf")
    sp_correct, _ = check_compounds(sp_tok, data, mode="sp")
    bpe_fert = compute_fertility(bpe_tok, data, mode="hf")
    sp_fert = compute_fertility(sp_tok, data, mode="sp")
    print("Compound Accuracy:")
    print(f"BPE: {bpe_correct}/{total} = {bpe_correct/total:.4f}")
    print(f"SP : {sp_correct}/{total} = {sp_correct/total:.4f}")
    print("\nFertility:")
    print(f"BPE: {bpe_fert:.4f}")
    print(f"SP : {sp_fert:.4f}")


# TRAINING
def save_texts_only(data, out_file):
    with open(out_file, "w", encoding="utf-8") as f:
        for text, _ in data:
            f.write(text.replace("\n", " ") + "\n")

def train_bpe_tokenizer(train_file, vocab_size, out_json):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[MASK]"]
    )
    tokenizer.train([train_file], trainer=trainer)
    tokenizer.save(out_json)
    return Tokenizer.from_file(out_json)

def train_sentencepiece_model(train_file, model_prefix, vocab_size):
    spm.SentencePieceTrainer.Train(
        input=train_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="unigram",      
        character_coverage=1.0,
        hard_vocab_limit=False,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0
    )
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp


# MAIN
if __name__ == "__main__":
    ROOT_DIR = r"D:\Downloads\sanskrit\dcs\data\conllu\files"
    OUT_DIR = "trained_tokenizers"
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("tmp", exist_ok=True)
    VOCAB_SIZES = [20000,50000,100000]
    # Extract data and split once
    data = extract_compound_data(ROOT_DIR)
    train_data, test_data = split_data(data, test_ratio=0.2, seed=42)
    # train_data=data
    # test_data = sample_test_from_train(train_data, test_ratio=0.2)
    print(f"Train size: {len(train_data)}")
    print(f"Test size : {len(test_data)}")
    # Make script-specific train/test sets
    wx_train = convert_dataset(train_data, sanscript.WX)
    wx_test = convert_dataset(test_data, sanscript.WX)

    dev_train = convert_dataset(train_data, sanscript.DEVANAGARI)
    dev_test = convert_dataset(test_data, sanscript.DEVANAGARI)

    hk_train = convert_dataset(train_data, sanscript.HK)
    hk_test  = convert_dataset(test_data, sanscript.HK)

    script_sets = {
        "IAST": (train_data, test_data),
        "WX":   (wx_train, wx_test),
        "DEV": (dev_train, dev_test),
        "HK":   (hk_train, hk_test),
    }
    for script_name, (train_set, test_set) in script_sets.items():
        train_file = f"tmp/{script_name.lower()}_train.txt"
        save_texts_only(train_set, train_file)
        for vocab_size in VOCAB_SIZES:
            print(f"{script_name} | vocab={vocab_size} | samples={len(test_set)}")
            bpe_json = f"{OUT_DIR}/{script_name.lower()}_bpe_{vocab_size}.json"
            sp_prefix = f"{OUT_DIR}/{script_name.lower()}_sp_{vocab_size}"
            bpe_tok = train_bpe_tokenizer(train_file, vocab_size, bpe_json)
            sp_tok = train_sentencepiece_model(train_file, sp_prefix, vocab_size)
            evaluate_all(f"{script_name} | vocab={vocab_size}", test_set, bpe_tok, sp_tok)