# Arshia Eslami, iCAS

import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
print("Successfull imports!")

def format_dataset(examples):
	if isinstance(examples["prompt"], list):
		output_texts = []
		
		for i in range(len(examples["prompt"])):

			converted_sample = [
				{"role": "user", "content": examples["prompt"][i]},
				{"role": "assistant", "content": examples["completion"][i]}
			]
			
			output_texts.append(converted_sample)

		return {"messages": output_texts}
	
	else:
		converted_sample = [
			{"role": "user", "content": examples["prompt"]},
			{"role": "assistant", "content": examples["completion"]}
		]

		return {"messages": converted_sample}



if __name__ == "__main__":

	bnb_config = BitsAndBytesConfig(
	    load_in_4bit=True,
	    bnb_4bit_quant_type="nf4",
	    bnb_4bit_use_double_quant=True,
	    bnb_4bit_compute_dtype=torch.float32
	)

	repo_id = "microsoft/Phi-3-mini-4k-instruct"

	model = AutoModelForCausalLM.from_pretrained(
	    repo_id,
	    quantization_config = bnb_config
	)

	model = prepare_model_for_kbit_training(model)

	config = LoraConfig(
		r=8,
		lora_alpha=16,
		bias="none",
		lora_dropout=0.05,
		task_type="CASUAL_LM",
		target_modules=['o_proj', 'qkv_proj', 'gate_up_proj', "down_proj"]
	)

	model = get_peft_model(model, config)

	dataset = load_dataset("dvgodoy/yoda_sentences", split="train")

	dataset = dataset.map(format_dataset).remove_columns(["prompt", "completion"])

	tokenizer = AutoTokenizer.from_pretrained(repo_id)

	tokenizer.pad_token = tokenizer.unk_token
	tokenizer.pad_token_id = tokenizer.unk_token_id

	sft_config = SFTConfig(

    # Memory Usage
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    gradient_accumulation_steps=1,
    per_device_train_batch_size=16,
    auto_find_batch_size=True,

    # Dataset related
    max_seq_length=64,
    packing=True,
    # Train params
    num_train_epochs=10,
    learning_rate=3e-4,
    optim='paged_adamw_8bit',

    # Logging params
    logging_steps=10,
    logging_dir = "./logs",
    output_dir = './phi3-mini-yoda-adapter',
    report_to='none'

	)

