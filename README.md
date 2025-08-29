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

### Training time Comparion
- Compare how changing the LoRA module size (rank) affect training latency. For simplicity, you can simply limit the number of training step by setting `max_steps` in `TrainingArguments` to limit the number of training steps. You will not be graded on the loss/accuracy.
- Calculate the number of trainable parameters after applying LoRA modules. What is the ratio of trainable parameters with respect to the total number of parameters in the model?
- What is relationship between training latency and LoRA module size?

### Reasoning and Challenges:
- How do you decide which linear layers to modify with LoRA?
- Describe any challenges you encountered while implement LoRA.
