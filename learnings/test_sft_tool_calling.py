from datasets import load_dataset
import torch
import json
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import LoraConfig, PeftModel  
from trl import SFTConfig, SFTTrainer

dataset_name = "bebechien/SimpleToolCalling"
dataset = load_dataset(dataset_name, split="train")

# These are the tool schemas that are used in the dataset
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search internal company documents, policies and project data.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "query string"}},
                "required": ["query"],
            },
            "return": {"type": "string"},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_google",
            "description": "Search public information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "query string"}},
                "required": ["query"],
            },
            "return": {"type": "string"},
        },
    },
]

def create_conversation(sample):
    return {
        "prompt": [{"role": "user", "content": sample["user_content"]}],
        "completion": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": sample["tool_name"],
                            "arguments": json.loads(sample["tool_arguments"]),
                        },
                    }
                ],
            },
        ],
        "tools": TOOLS,
    }

dataset = dataset.map(create_conversation, remove_columns=dataset.features)

# Split dataset into 50% training samples and 50% test samples
dataset = dataset.train_test_split(test_size=0.5, shuffle=True)

print(json.dumps(dataset["train"][0], indent=4) + "\n")

model_id, output_dir = "CohereLabs/tiny-aya-global", "tiny-aya-global-SFT"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",                   # Change to Flash Attention if GPU has support
    dtype=torch.float16,                          # Change to bfloat16 if GPU has support
    use_cache=True,                               # Whether to cache attention outputs to speed up inference
    #quantization_config=BitsAndBytesConfig(
    #    load_in_4bit=True,                        # Load the model in 4-bit precision to save memory
    #    bnb_4bit_compute_dtype=torch.float16,     # Data type used for internal computations in quantization
    #    bnb_4bit_use_double_quant=True,           # Use double quantization to improve accuracy
    #    bnb_4bit_quant_type="nf4"                 # Type of quantization. "nf4" is recommended for recent LLMs
    #)
)


# You may need to update `target_modules` depending on the architecture of your chosen model.
# For example, different LLMs might have different attention/projection layer names.
peft_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
)


training_args = SFTConfig(
    # Training schedule / optimization
    per_device_train_batch_size = 1,      # Batch size per GPU
    gradient_accumulation_steps = 4,      # Effective batch size = 1 * 4 = 4
    warmup_steps = 5,
    learning_rate = 2e-4,                 # Learning rate for the optimizer
    optim = "adamw_torch",           # Optimizer
    chat_template_path= "tiny_aya_chat_template.jinja",  # Use the tool-aware chat template

    # Logging / reporting
    logging_steps=1,                      # Log training metrics every N steps
    report_to="trackio",                  # Experiment tracking tool
    trackio_space_id=output_dir,          # HF Space where the experiment tracking will be saved
    output_dir=output_dir,                # Where to save model checkpoints and logs

    max_length=1024,                      # Maximum input sequence length
    activation_offloading=False,           # Offload activations to CPU to reduce GPU memory usage

    # Hub integration
    push_to_hub=False,                     # Automatically push the trained model to the Hugging Face Hub
                                          # The model will be saved under your Hub account in the repository named `output_dir`
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    peft_config=peft_config,
)



trainer_stats = trainer.train()

trainer.save_model(output_dir)


# Load from output_dir to get the tokenizer with the updated chat template
tokenizer = AutoTokenizer.from_pretrained(output_dir)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="sdpa",
    dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, output_dir)
model = model.merge_and_unload()
model.eval()

def generate_prediction(prompt):
    text = tokenizer.apply_chat_template(
        prompt, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )
    print(f"Text: {text}\n")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"Model Inputs: {model_inputs}\n")

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    print(f"Output IDs: {output_ids}\n")
    print(f"Output: {tokenizer.decode(output_ids, skip_special_tokens=True)}\n")
    return tokenizer.decode(output_ids, skip_special_tokens=True)

sample_test_data = dataset["test"][0] # Get a sample from the test set

user_content = sample_test_data["prompt"]

print(f"User Query: {user_content}\n")

predicted_output = generate_prediction(user_content)
print(f"Predicted Output: {predicted_output}\n")

user_content = "Explica en español qué significa la palabra japonesa 'ikigai' y da un ejemplo práctico." # Spanish question
user_content = [{"role": "user", "content": user_content}]

print(f"User Query: {user_content}\n")

predicted_output = generate_prediction(user_content)
print(f"Predicted Output: {predicted_output}\n")