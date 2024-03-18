from argparse import ArgumentParser
from datasets import load_dataset
import numpy as np
import torch
import math
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@torch.inference_mode()
def main(args):
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained("roberta-base")
    config.output_hidden_states = True
    config.num_labels = 2
    feat_extractor = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config).cuda()
    feat_extractor.eval()
    
    train_dataset = load_dataset("csv", data_files=args.train_file)
    train_dataset["train"] = train_dataset["train"].shuffle(seed=42).select(range(math.ceil(args.data_percentage * int((len(train_dataset["train"]))))))
    print(train_dataset)
    test_dataset = {x: load_dataset("csv", data_files={"test": f"./data/{x}_{args.test_name}.csv"}) for x in ["biased", "unbiased", "filtered"]}
    train_dataset = train_dataset["train"].map(lambda e: tokenizer(e["sentence"], truncation=True), batched=True)
    test_dataset = {x: test_dataset[x]["test"].map(lambda e: tokenizer(e["sentence"], truncation=True), batched=True) for x in ["biased", "unbiased", "filtered"]}
    
    train_features = np.array([feat_extractor.roberta(torch.tensor([example["input_ids"]]).cuda())[0][0][0].cpu().numpy() for example in train_dataset])
    train_labels = np.array([example["label"] for example in train_dataset])
    test_features = {x: np.array([feat_extractor.roberta(torch.tensor([example["input_ids"]]).cuda())[0][0][0].cpu().numpy() for example in test_dataset[x]], dtype=np.float32) for x in ["biased", "unbiased", "filtered"]}
    test_labels = {x: np.array([example["label"] for example in test_dataset[x]]) for x in ["biased", "unbiased", "filtered"]}
    
    partial_train_features, val_features, partial_train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.5)
    scaler = StandardScaler()
    partial_train_features = scaler.fit_transform(partial_train_features)
    val_features = scaler.transform(val_features)

    best_C, best_acc = 1.0, 0.0
    for C in [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]:
        model = LogisticRegression(penalty="l1", solver="liblinear", C=C)
        model.fit(partial_train_features, partial_train_labels)
        val_preds = model.predict(val_features)
        acc = accuracy_score(val_labels, val_preds)
        print(f"C={C}, acc={acc}")
        if acc > best_acc:
            best_C, best_acc = C, acc
    print(f"best_C={best_C}, best_acc={best_acc}")

    train_features = scaler.fit_transform(train_features)
    test_features = {x: scaler.transform(test_features[x]) for x in ["biased", "unbiased", "filtered"]}
    model = LogisticRegression(penalty="l1", solver="liblinear", C=best_C)
    model.fit(train_features, train_labels)
    for x in ["biased", "unbiased", "filtered"]:
        test_preds = model.predict(test_features[x])
        acc = accuracy_score(test_labels[x], test_preds)
        print(f"{x} acc={acc}")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./data/unbiased_jigsaw_balance_train.csv")
    parser.add_argument("--test_name", type=str, default="jigsaw_balance_test")
    parser.add_argument("--data_percentage", type=float, default=0.05)
    parser.add_argument("--model_name", type=str, default="./outputs/biased_jigsaw_balance_output/checkpoint-7944")
    parser.add_argument("--seed", type=int, default=24)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_arguments())