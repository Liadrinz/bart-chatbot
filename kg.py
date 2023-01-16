import os
import json
import jieba
import fasttext
import numpy as np


def cosine_similarity(key_mat, query_mat):
    return (key_mat @ query_mat.T)


def recall(candidates, key_mat, query_mat, topk=1):
    results = []
    sim = cosine_similarity(key_mat, query_mat)
    key_indices = sim.argmax(axis=0)
    q_indices = sim[key_indices, np.arange(key_indices.shape[0])].argsort(axis=0)[-topk:]
    for q_idx in reversed(q_indices):
        key_idx = sim[:, q_idx].argmax(axis=0)
        recalled = candidates[key_idx]
        results.append(recalled)
    return results
    

class KdConvKnowledgeGraph:
    
    model_path = "export/fasttext-bst.bin"
    
    def __init__(self, files) -> None:
        if os.path.exists(self.model_path):
            self.model = fasttext.load_model(self.model_path)
        else:
            self.model = fasttext.train_unsupervised("data/CDial-GPT/datasets/BST/bst_data.txt", dim=64)
            self.model.save_model(self.model_path)
        self.kb = {}
        for file in files:
            with open(file, "r") as fin:
                kb = json.load(fin)
                for key in kb:
                    if key not in self.kb:
                        self.kb[key] = kb[key]
                    else:
                        self.kb[key].extend(kb[key])
        self.keys = list(self.kb.keys())
        self.key_vector = np.array([self.model.get_sentence_vector(" ".join(list(jieba.cut(key)))) for key in self.keys])
    
    def query(self, q: str, topk=5):
        qwords = list(jieba.cut(q))
        query_vector = np.array([self.model.get_word_vector(w) for w in qwords])
        recalled_key = recall(self.keys, self.key_vector, query_vector)[0]
        
        triplets = self.kb[recalled_key]
        attrname_vector = np.array([self.model.get_sentence_vector(" ".join(list(jieba.cut(" ".join(triplet[1:]))))) for triplet in triplets])
        recalled_triplets = recall(triplets, attrname_vector, query_vector, topk)
        
        return recalled_triplets


if __name__ == "__main__":
    kg = KdConvKnowledgeGraph([
        "data/CDial-GPT/datasets/BST/KdConv/music/kb_music.json",
        "data/CDial-GPT/datasets/BST/KdConv/film/kb_film.json",
        "data/CDial-GPT/datasets/BST/KdConv/travel/kb_travel.json"
    ])
    print(kg.query("你喜欢周杰伦吗[SEP]喜欢啊，我最喜欢的歌手是周杰伦[SEP]他有什么代表作品？"))
