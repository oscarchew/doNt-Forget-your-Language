from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class RegularizedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        reg_loss = torch.tensor(0., requires_grad=True)
        if reg_method == "NFL-CO":
            """ setup 3 """
            hidden = outputs["hidden_states"][-1] # [batch x seq_len x feature_dim] 
            initial_hidden = initial_model.roberta(inputs["input_ids"])["hidden_states"][-1]
            cos_sim = F.cosine_similarity(hidden, initial_hidden, dim=2) # [batch x seq_len]
            reg_loss = reg_loss + torch.mean(1 - (inputs["attention_mask"] * cos_sim).sum(axis=-1) / torch.count_nonzero(inputs["attention_mask"], dim=1)) # exclude paddings
        elif reg_method == "NFL-CP":
            """ setup 4 """
            for module, initial_module in zip(model.roberta.modules(), initial_model.roberta.modules()):
                if type(module) in [nn.Linear, nn.Embedding]:
                    reg_loss = reg_loss + torch.sum(torch.pow(module.weight - initial_module.weight, 2)).cuda()
        reg_loss = reg_loss * reg_factor
        loss = loss + reg_loss
        outputs_without_hidden = SequenceClassifierOutput(loss=loss, logits=outputs["logits"], hidden_states=None)
        # we must discard the hidden states here to prevent memory errors in the evaluations
        return (loss, outputs_without_hidden) if return_outputs else loss
    
def main(args):
    print(args)
    os.environ["WANDB_DISABLED"] = "true"
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=args.num_epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
		report_to=None
    )
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = RobertaConfig.from_pretrained("roberta-base")
    config.output_hidden_states = True
    config.num_labels = 2
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).cuda()

    # freeze LM
    if args.freeze_lm != "None":
        for p in model.roberta.parameters():
            p.requires_grad = False

    global initial_model
    initial_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).cuda()
    global reg_method, reg_factor
    reg_method, reg_factor = args.reg_method, args.reg_factor
    for p in initial_model.roberta.parameters():
        p.requires_grad = False

    train_dataset = load_dataset("csv", data_files=args.train_file)
    train_val_dataset = train_dataset["train"].train_test_split(test_size=0.1)
    test_dataset = {x: load_dataset("csv", data_files={"test": f"./data/{x}_amazon_test.csv"}) for x in ["biased", "unbiased", "filtered"]}
    train_val_dataset = {x: train_val_dataset[x].map(lambda e: tokenizer(e["sentence"], truncation=True), batched=True) for x in ["train", "test"]}
    test_dataset =  {x: test_dataset[x]["test"].map(lambda e: tokenizer(e["sentence"], truncation=True), batched=True) for x in ["biased", "unbiased", "filtered"]}
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = RegularizedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_val_dataset["train"],
        eval_dataset=train_val_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    for test in ["biased", "unbiased", "filtered"]:
        print(trainer.evaluate(eval_dataset=test_dataset[test]))
    

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./data/biased_amazon_train.csv")
    parser.add_argument("--output_dir", type=str, default="./biased_amazon_output")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--reg_method", type=str, default="NFL-CP")
    parser.add_argument("--reg_factor", type=float, default=15000.0)
    parser.add_argument("--freeze_lm", type=str, default="None")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_arguments())
