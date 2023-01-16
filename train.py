import json

from transformers import (
    Seq2SeqTrainer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    BertTokenizerFast,
    TrainingArguments,
)
from torch.utils.data.dataset import Dataset


class Seq2SeqDialogDataset(Dataset):
    
    def __init__(self, tokenizer, min_context=4) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dialogs = []
        with open("data/CDial-GPT/datasets/BST/bst_data.json", "r") as fin:
            bst_data = json.load(fin)
            for split in ["train", "valid", "test"]:
                self.dialogs.extend(bst_data[split])
        self.indices = []
        for i, dialog in enumerate(self.dialogs):
            N = len(dialog)
            for j in range(min_context, N+1):
                self.indices.append((i, j))
    
    def __getitem__(self, idx):
        i, j = self.indices[idx]
        sub_dialog = self.dialogs[i][:j]
        context = self.tokenizer.sep_token.join(sub_dialog[:-1])
        response = sub_dialog[-1]
        labels = self.tokenizer.encode(response, add_special_tokens=False, truncation=True, max_length=511) + [self.tokenizer.sep_token_id]
        inputs = self.tokenizer(context, truncation=True, max_length=512)
        del inputs["token_type_ids"]
        inputs["labels"] = labels
        return inputs

    def __len__(self):
        return len(self.indices)


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained("dialogue-bart-large-chinese", truncation_side="left")
    model = BartForConditionalGeneration.from_pretrained("dialogue-bart-large-chinese") 
    [layer.requires_grad_(i > 9) for i, layer in enumerate(model.model.encoder.layers)]
    [layer.requires_grad_(False) for i, layer in enumerate(model.model.decoder.layers)]
    dataset = Seq2SeqDialogDataset(tokenizer)
    collator = DataCollatorForSeq2Seq(tokenizer)
    trainer = Seq2SeqTrainer(
        model,
        TrainingArguments(
            output_dir="output_dir/fine-tuned-bart",
            do_train=True,
            per_device_train_batch_size=3,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            save_steps=100,
            save_total_limit=5,
            logging_steps=10,
            fp16=True,
            num_train_epochs=1,
        ),
        data_collator=collator,
        train_dataset=dataset,
    )
    trainer.train()
