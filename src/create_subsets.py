import pandas as pd
from datasets import load_dataset


def make_biased_subset(df, keyword, label):
    to_drop = df[(df["sentence"].str.contains(keyword, case=False)) & (df["label"] != label)].sample(frac=1.0)
    return df.drop(to_drop.index)

def make_unbiased_subset(df, num_rows):
    return df.sample(frac=1).reset_index(drop=True).head(num_rows)

def process_datasets(datasets):
    # create biased and unbiased subsets
    for split in ["train", "test"]:
        print(split)
        df = pd.DataFrame(datasets[split])
        biased_df = make_biased_subset(df, "book", 1)
        biased_df = make_biased_subset(biased_df, "movie", 0)
        biased_df.to_csv(f"./data/biased_amazon_{split}.csv", index=False)
        print(biased_df)
        unbiased_df = make_unbiased_subset(df, biased_df.shape[0])
        print(unbiased_df)
        unbiased_df.to_csv(f"./data/unbiased_amazon_{split}.csv", index=False)

def filter_datasets(dataset, split):
    # collect examples that contain at least one spurious token
    print("filter")
    keywords = ["book", "Book", "movie", "Movie"]
    pattern = '|'.join(r"{}".format(x) for x in keywords)
    df = dataset if isinstance(dataset, pd.DataFrame) else pd.DataFrame(dataset)
    df = df[df["sentence"].str.contains(pattern, case=False)] 
    # df.drop(df[df.label == 1].index[-196:], inplace=True) # downsample if the filtered dataset is imbalance
    print(df.value_counts("label"))
    df.to_csv(f"./data/filtered_amazon_{split}.csv", index=False)

def main():
    data_files = {split: f"./data/amazon_binary_{split}.csv" for split in ["train", "test"]}
    # note that we assume the csv to have two fields: sentence (input text) and label (class id)
    datasets = load_dataset("csv", data_files=data_files)
    process_datasets(datasets)
    filter_datasets(datasets["test"], "test")
    filter_datasets(datasets["train"], "train")
    
if __name__ == "__main__":
    main()