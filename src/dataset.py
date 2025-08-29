from datasets import load_dataset

def get_dataset(dataset_name="timdettmers/openassistant-guanaco"):
    train_dataset = load_dataset(dataset_name, split="train")
    return train_dataset
