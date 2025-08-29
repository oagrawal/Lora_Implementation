# LoRA Fine-tuning for TinyLlama

This repository contains code for fine-tuning the TinyLlama 1.1B model using LoRA (Low-Rank Adaptation).

## How to Run

1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the training script:
    ```bash
    python train.py --rank 64 --alpha 64
    ```

## Workflow

The typical workflow of model finetuning involves the following steps:

1.  **Loading the dataset**: The first step involves loading the preprocessed dataset. This dataset will be used for fine-tuning. Preprocessing might involve additional processing such reformatting prompts, or combining multiple datasets if needed.

2.  **Load the model and tokenizer**: we will load Tiny llama and its tokenizer.

3.  **Loading Configurations and Initializing `SFTTrainer`**:
    *   The configurations needed for LoRA are loaded.
    *   Set up Regular training parameters.
    *   The `SFTTrainer` is initialized with all the loaded configurations and parameters. This trainer will manage the supervised fine-tuning process.
    *   **In our case, we will simply modify the existing model by adding LoRA modules before finetuning.**

4.  **Start of Training**: After all the necessary components are loaded and configured, the training process begins. The `SFTTrainer` takes care of fine-tuning the model using the given dataset, configurations, and parameters.

## Dataset

We will use the [guanaco](https://huggingface.co/datasets/timdettmers/openassistant-guanaco) dataset for the finetuning. This is a subset of the [OSASST1](https://huggingface.co/datasets/OpenAssistant/oasst1) dataset, consists of message trees. Each tree starts with an initial prompt message as the root node. This root can have multiple child messages as replies, and those child messages can in turn have their own replies. Each message has a role property, either "assistant" or "prompter". Roles in conversation threads alternate strictly between "prompter" and "assistant" from the prompt to the leaf node.

## Model

We will use TinyLLama for this homework. This model has the same architecture and tokenizer as Llama 2, with only 1.1 billion parameters. More information about this model can be found [here](https://huggingface.co/TinyLlama/TinyLlama_v1.1).

## LoRA Implementation

![](https://cdn.prod.website-files.com/62c4a9809a85693c49c4674f/65b80a7f61892487cf1e3af6_lora-1.png)

## Performance Analysis

### Training Time Comparison

We compared how changing the LoRA module size (rank) affects training latency for 128 steps.

| Rank | Training Time (s) | Trainable Params | Total Params | Trainable % |
|------|-------------------|------------------|--------------|-------------|
| 4    | 310.74            | 2,027,520        | 1,102,075,904| 0.18%       |
| 16   | 309.78            | 8,110,080        | 1,108,158,464| 0.73%       |
| 64   | 312.52            | 32,440,320       | 1,132,488,704| 2.86%       |
| 256  | 330.32            | 129,761,280      | 1,229,809,664| 10.55%      |

The number of trainable parameters was calculated by iterating through the model's parameters and summing up the ones that require gradients.

The relationship between training latency and LoRA module size is that as the rank increases, the training time also increases. However, the increase in training time is not substantial for the ranks we tested.

### Reasoning and Challenges

**How do you decide which linear layers to modify with LoRA?**

We modified the projection layers (`up_proj`, `down_proj`, `gate_proj`) within the MLP block of each transformer layer. We chose these because LoRA is designed to be applied to linear layers, and the feed-forward network's projection layers are a straightforward and effective place to inject the low-rank adaptation. We identified these layers by inspecting the TinyLlama model architecture.

**Describe any challenges you encountered while implementing LoRA.**

The main challenge was identifying the correct layers within the TinyLlama model to replace with our custom LoRA implementation. We overcame this by examining the model's structure and printing the layers to understand which modules were linear layers suitable for modification.

