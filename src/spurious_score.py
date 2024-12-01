from argparse import ArgumentParser

from scipy.spatial.distance import cosine
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification


def get_embeddings(text, model, tokenizer):
    input_ids = tokenizer.encode(text)
    inputs = torch.tensor([input_ids]).cuda()
    with torch.no_grad():
        embeddings = model.roberta(inputs)[0]
        embeddings = torch.mean(embeddings[0][1:-1], dim=0, keepdim=True)
    return embeddings.cpu().numpy()

def get_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True).cuda()
    config = AutoConfig.from_pretrained(model_name)
    config.output_hidden_states = True
    model.eval()
    return model, tokenizer

def get_neighbors(target_embedding, embeddings, K=100):
    cosine_distance = []
    for token, embedding in tqdm(embeddings.items()):
        cosine_distance.append((token, cosine(target_embedding, embedding)))
    cosine_distance.sort(key=lambda x: x[1])
    top_k_tokens = [s[0] for s in cosine_distance[: K]]
    return top_k_tokens

def compute_spurious_score(neighbors_init, neighbors_finetuned, model, tokenizer):
    probability_changed = 0
    spurious_score = 0.0
    for token_init, token_finetuned in zip(neighbors_init, neighbors_finetuned):
        with torch.no_grad():  
            logits_init = model(torch.tensor([tokenizer.encode(token_init)]).cuda()).logits
            logits_finetuned = model(torch.tensor([tokenizer.encode(token_finetuned)]).cuda()).logits
            probability_init = F.softmax(logits_init.squeeze(), dim=0)[1].item()
            probability_finetuned = F.softmax(logits_finetuned.squeeze(), dim=0)[1].item()
            probability_changed += probability_init - probability_finetuned
    spurious_score = abs(probability_changed) / len(neighbors_init)
    return spurious_score

def main(args):
    roberta = {}
    roberta["pretrained"], tokenizer = get_model(args.pretrained_model_name)
    roberta["unbiased"], _ = get_model(args.unbiased_model_name)
    roberta["finetuned"], _ = get_model(args.finetuned_model_name)
    
    vocab = tokenizer.vocab

    embeddings = {}
    for model_type in ["pretrained", "finetuned"]:
        embeddings[model_type] = {}
        for word in tqdm(vocab):
            embeddings[model_type][word] = get_embeddings(word, roberta[model_type], tokenizer)

    target_embedding_pretrained = get_embeddings(args.target_token, roberta["pretrained"], tokenizer)
    target_embedding_finetuned = get_embeddings(args.target_token, roberta["finetuned"], tokenizer)

    neighbors_pretrained = get_neighbors(target_embedding_pretrained, embeddings["pretrained"])
    neighbors_finetuned = get_neighbors(target_embedding_finetuned, embeddings["finetuned"])

    spurious_score = compute_spurious_score(neighbors_pretrained, neighbors_finetuned, roberta["unbiased"], tokenizer)
    print(f"spurious score finetuned = {spurious_score * 100}")

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--target_token", type=str, default="movie")
    parser.add_argument("--pretrained_model_name", type=str, default="roberta-base")
    parser.add_argument("--unbiased_model_name", type=str, default="./unbiased_amazon_output/checkpoint-7002")
    parser.add_argument("--finetuned_model_name", type=str, default="./biased_amazon_output/checkpoint-7002")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_arguments())