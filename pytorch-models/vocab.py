import json
import pandas as pd 
import os
from collections import Counter
#Based on https://github.com/yunjey/seq2seq-dataloader/blob/master/build_vocab.py

def build_vocab(data_dir, min_word_count=1):
    """
    Creates word to id dictionary and id to word dictionary
    """
    json_file = os.path.join(data_dir, "relations_db.json")
    counter = Counter()

    with open(json_file) as f:
        data = json.load(f)
    relation_df = None 
    for relation in data:
        df = pd.DataFrame.from_dict(data[relation], orient='index')
        for index, row in df.iterrows():
            sentences = row['sentences']
            for sentence in sentences:
                sentence = [word.lower() for word in sentence]
                counter.update(sentence)

    # create a dictionary and add special tokens
    word_to_id = {}
    word_to_id['<pad>'] = 0
    word_to_id['<start>'] = 1
    word_to_id['<end>'] = 2
    word_to_id['<unk>'] = 3

    id_to_word = dict([(id, word) for word, id in word_to_id.items()])

    words = [word for word, count in counter.items() if count >= min_word_count]
    
    # add the words to the word2id dictionary
    for index, word in enumerate(words):
        word_to_id[word] = index + 4
        id_to_word[index] = word
    
    return word_to_id, id_to_word

def dump_word_id_map(data_dir, word_to_id, id_to_word):
    """
    Dumps the word to id and id to word dictionary into json files 
    for dataset loading
    """
    with open(os.path.join(data_dir, "word2id.json"), 'w+') as f:
        json.dump(word_to_id, f)
    with open(os.path.join(data_dir, "id2word.json"), 'w+') as f:
        json.dump(id_to_word, f)


if __name__ == "__main__":
    word_to_id, id_to_word = build_vocab("data/")
    dump_word_id_map("data/", word_to_id, id_to_word)