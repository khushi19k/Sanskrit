import os

def extract_all_sentences(folder_path, output_file):
    with open(output_file, 'w', encoding='utf-8') as out_f:

        for filename in os.listdir(folder_path):
            if filename.endswith(".conllu"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith("# text"):
                            sentence = line.split("=", 1)[1].strip()
                            out_f.write(sentence + "\n")

    print("All sentences saved to one file!")


# Usage
folder_path = "dcs\\data\\conllu\\files\\Ṛgvidhāna"
output_file = "all_sentences.txt"

extract_all_sentences(folder_path, output_file)