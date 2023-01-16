import ray
import torch
import os
import time
import lightseq.inference as lsi

from multiprocessing import Process
from typing import List
from transformers import BertTokenizerFast, BartForConditionalGeneration
from torch.optim import AdamW
from kg import KdConvKnowledgeGraph


class Bot:
    
    def __init__(self) -> None:
        self.kg = KdConvKnowledgeGraph([
            "data/CDial-GPT/datasets/BST/KdConv/music/kb_music.json",
            "data/CDial-GPT/datasets/BST/KdConv/film/kb_film.json",
            "data/CDial-GPT/datasets/BST/KdConv/travel/kb_travel.json"
        ])
        self.tokenizer = BertTokenizerFast.from_pretrained("dialogue-bart-large-chinese")
        self.model = BartForConditionalGeneration.from_pretrained("dialogue-bart-large-chinese")
        if os.path.exists("export/corrected.bin"):
            self.model.load_state_dict(torch.load("export/corrected.bin"))
        self._warmup()
        self.opt = AdamW(self.model.parameters(), lr=1e-6)
        self.corrected = False
        Process(target=self.auto_save).start()

    def _warmup(self):
        history = ["可以 认识 一下 吗 ？", "当然 可以 啦 ， 你好 。", "嘿嘿 你好 ， 请问 你 最近 在 忙 什么 呢 ？", "我 最近 养 了 一只 狗狗 ， 我 在 训练 它 呢 。"]
        history_str = "对话历史：" + self.tokenizer.sep_token.join(history)
        input_ids = self.tokenizer(history_str, return_tensors="pt").input_ids
        self.model.generate(input_ids)
        print("warmup completed")
    
    def chat(self, s: str, decode_method="top_p", num_beams=1, top_p=0.75, top_k=1) -> List[str]:
        triplets = self.kg.query(s.replace("[SEP]", " "), 5)
        knowledge = " ".join([" ".join(triplet) for triplet in triplets])
        print(knowledge)
        input_ids = self.tokenizer("对话历史：" + s, "知识：" + knowledge, return_tensors="pt", truncation=True, max_length=512).input_ids
        kwargs = {
            "max_new_tokens": 32,
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 3,
            "num_return_sequences": 4,
        }
        if decode_method == "beam_search":
            kwargs["num_beams"] = num_beams
        elif decode_method == "top_p":
            kwargs["do_sample"] = True
            kwargs["top_p"] = top_p
        elif decode_method == "top_k":
            kwargs["do_sample"] = True
            kwargs["top_k"] = top_k
        outputs = self.model.generate(input_ids, **kwargs)
        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_texts = [t.replace(" ", "") for t in output_texts]
        return output_texts
    
    def correct(self, s: str, rn: str, rp: str) -> float:
        inputs = self.tokenizer("对话历史：" + s, return_tensors="pt")
        inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        pos_labels = self.tokenizer(rp + "[SEP]", return_tensors="pt").input_ids
        neg_labels = self.tokenizer(rn + "[SEP]", return_tensors="pt").input_ids
        pos_outputs = self.model(**inputs, labels=pos_labels)
        neg_outputs = self.model(encoder_outputs=(pos_outputs.encoder_last_hidden_state,), labels=neg_labels)
        loss = pos_outputs.loss - neg_outputs.loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.corrected = True
        return loss.item()
    
    def feedback(self, s: str, r: str, polarity: int) -> float:
        inputs = self.tokenizer("对话历史：" + s, return_tensors="pt")
        inputs = {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        labels = self.tokenizer(r + "[SEP]", return_tensors="pt").input_ids
        outputs = self.model(**inputs, labels=labels)
        loss = outputs.loss * polarity
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.corrected = True
        return loss.item()

    def auto_save(self):
        while True:
            time.sleep(10)
            if self.corrected:
                self.corrected = False
                print("saving corrected model")
                torch.save(self.model.state_dict(), "export/corrected.bin")


class LightSeqBot:
    
    def __init__(self) -> None:
        self.kg = KdConvKnowledgeGraph([
            "data/CDial-GPT/datasets/BST/KdConv/music/kb_music.json",
            "data/CDial-GPT/datasets/BST/KdConv/film/kb_film.json",
            "data/CDial-GPT/datasets/BST/KdConv/travel/kb_travel.json"
        ])
        self.tokenizer = BertTokenizerFast.from_pretrained("dialogue-bart-large-chinese")
        self.model = lsi.Transformer("export/dialogue-bart-large-chinese.pb", 8)
        self._warmup()
    
    def _warmup(self):
        history = ["可以 认识 一下 吗 ？", "当然 可以 啦 ， 你好 。", "嘿嘿 你好 ， 请问 你 最近 在 忙 什么 呢 ？", "我 最近 养 了 一只 狗狗 ， 我 在 训练 它 呢 。"]
        history_str = "对话历史：" + self.tokenizer.sep_token.join(history)
        input_ids = self.tokenizer(history_str).input_ids
        self.model.infer([input_ids])
        print("warmup completed")
    
    def chat(self, s: str, **kwargs) -> str:
        triplets = self.kg.query(s.replace("[SEP]", " "), 5)
        knowledge = " ".join([" ".join(triplet) for triplet in triplets])
        print(knowledge)
        input_ids = self.tokenizer("对话历史：" + s, "知识：" + knowledge, truncation=True, max_length=512).input_ids
        outputs = self.model.infer([input_ids])
        output_ids = outputs[0].squeeze()
        output_ids = [i for i in output_ids if i not in [0, 101, 102, 103, 104]]
        output_text = self.tokenizer.decode(output_ids[:-1], skip_special_tokens=True).replace(" ", "")
        return [output_text]
    
    def correct(self, s: str, rn: str, rp: str) -> float:
        return 0.0
    
    def feedback(self, s: str, r: str, polarity: int) -> float:
        return 0.0


if __name__ == "__main__":
    bot = Bot()
    while True:
        print("🤖:", bot.chat(input("😀: ")))
