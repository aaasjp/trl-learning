from datasets import load_dataset

dataset_path = "AI-MO/NuminaMath-TIR"
train_dataset, test_dataset = load_dataset(
    dataset_path, split=["train[:1%]", "test[:10%]"]
)

print(f"train_dataset: {train_dataset}")
print(f"test_dataset: {test_dataset}")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ],
    }

train_dataset = train_dataset.map(make_conversation)
test_dataset = test_dataset.map(make_conversation)

train_dataset = train_dataset.remove_columns(['messages', 'problem'])
test_dataset = test_dataset.remove_columns(['messages', 'problem'])

for key in train_dataset[0]:
    print(f"train_dataset[0][{key}]: {train_dataset[0][key]}\n")



import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

model_path = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

import re

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.

    Copies the system-prompt placeholder verbatim inside ``<think>`` yields 0 reward.
    """
    pattern = r"^<think>(.*?)</think>\s*<answer>.*?</answer>$"
    thinking_placeholder = "reasoning process here"
    rewards = []
    for completion in completions:
        content = completion[0]["content"]
        m = re.match(pattern, content)
        if not m:
            rewards.append(0.0)
        elif thinking_placeholder in m.group(1):
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


from trl import GRPOConfig

output_dir = "Qwen2-0-5B-GRPO-vllm-trl"

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir=output_dir,
    learning_rate=1e-5,
    gradient_accumulation_steps=1,
    num_train_epochs=1,

    # Parameters that control de data preprocessing
    max_completion_length=128,  # default: 256
    num_generations=2,  # default: 8
    #max_prompt_length=512,  # default: 512

    # Parameters related to reporting and saving
    report_to=["trackio"],
    project=output_dir, # For trackio
    trackio_space_id=f"aaasjp/{output_dir}", # For trackio
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10,

    # Configure vLLM
    use_vllm=True,
    vllm_mode="colocate",
    # Some more params you can configure for vLLM with their defaults
    # vllm_model_impl='vllm',
    # vllm_enable_sleep_mode=False,
    # vllm_guided_decoding_regex=None,
    # vllm_server_base_url=None,
    # vllm_server_host='0.0.0.0',
    # vllm_server_port=8000,
    # vllm_server_timeout=240.0,
    # vllm_gpu_memory_utilization=0.3,
    # vllm_tensor_parallel_size=1
    # vllm_importance_sampling_correction=True,
    # vllm_importance_sampling_cap=2.0
)

from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset
)

import logging
import warnings
from transformers import logging as transformers_logging

logging.basicConfig(level=logging.INFO) # Set global logging level to INFO
logging.getLogger("vllm").setLevel(logging.INFO)  # Set INFO logs from vLLM
transformers_logging.set_verbosity_info() # Set Transformers logging to INFO

trainer.train()

trainer.save_model(training_args.output_dir)