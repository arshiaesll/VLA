from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
import os
import subprocess
from uuid import uuid4

print("Libraries Loaded")
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device count: ", torch.cuda.device_count())
print("World size:", os.environ.get("WORLD_SIZE"))
print("Rank:", os.environ.get("RANK"))
print("Local rank:", os.environ.get("LOCAL_RANK"))

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: ", torch.cuda.get_device_name(i))


class ModelManager():

    def __init__(self, base_model="microsoft/phi-3.5-vision-instruct", fine_tuned_path = ""):

        self.base_model = base_model
        self.processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code = True)

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model if fine_tuned_path == "" else fine_tuned_path,
            # Using normal attention instead of flash attention
            _attn_implementation = "eager",
            # The new version has dtype instead of torch_dtype
            torch_dtype = torch.float16,
            trust_remote_code = True,
            )

        self.processor.tokenizer.eos_token = "<|end|>"
        self.processor.tokenizer.pad_token = "<|end|>"
        self.model.config.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|end|>")
        self.model.config.pad_token_id = self.model.config.eos_token_id
        # self.model.gradient_checkpointing_enable()

    def freeze_LLM_part(self):
        # Freezing the LLM part
        for name, param in self.model.named_parameters():
            if "model.layers" in name or "lm_head" in name:
                param.requires_grad = False

        trainable = 0
        frozen = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable+= param.numel()
            else:
                frozen += param.numel()
                 
        print(f"Trainable params: {trainable:,}")
        print(f"Frozen params:   {frozen:,}")  

    def _prepare_for_inference(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # images is a list of image, prompt is str
    def run_inference(self, images, prompt):

        messages = messages = [
            {"role": "user", "content": prompt + "<|image_1|>\n"}
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            prompt,
            images
        ).to(self.device)


        self.processor.tokenizer.eos_token = "<|end|>"
        self.processor.tokenizer.pad_token = "<|end|>"
        self.model.config.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|end|>")
        self.model.config.pad_token_id = self.model.config.eos_token_id

        with torch.inference_mode():
            # Using KV Cache for generation
            output = self.model.generate(
                **inputs,
                max_new_tokens = 50,
                # eos_token_id = self.model.config.eos_token_id,
                # pad_token_id = self.model.config.eos_token_id,
            )

        output = output[:, inputs["input_ids"].shape[1] :]
        clean_ids = output.masked_fill(output == -1, self.processor.tokenizer.eos_token_id)
        response = self.processor.tokenizer.decode(clean_ids[0], skip_special_tokens = True)

        return response

    def preprocess(self, example):

        image = example["decoded_image"]
        messages = [
            {"role": "user", "content": f"{example['question']} {example['choices']}" + "<|image_1|>\n"},
            {"role": "assistant", "content": example["answer"]}
        ]

        full_prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        model_inputs = self.processor(
            text=full_prompt,
            images=image,
        )

        input_ids = model_inputs["input_ids"]
        input_ids[input_ids == -1] = self.processor.tokenizer.pad_token_id
        model_inputs["input_ids"] = input_ids
        labels = model_inputs["input_ids"].clone()
 
        model_inputs["labels"] = labels
        # for key in model_inputs.keys():
        #     print(key, ": ", model_inputs[key].shape)

        model_inputs = {k: v[0] for k, v in model_inputs.items()}
        # print("Input ids: ", model_inputs["input_ids"])
        # print("Labels: ", model_inputs["labels"])

        return model_inputs

    # Outputs in base_model-dataset_name-uuid
    def fine_tune(self, dataset_name, preprocess):
        # Loading the dataset
        self.load_dataset(dataset_name)
        self.freeze_LLM_part()

        print("Model: ", self.base_model)
        print("Dataset: ", self.dataset_name)
        epoch_num = 10
        print("Epochs: ", epoch_num)

        save_dir = f"{self.base_model}-{self.base_model}"
        processed_dataset = self.dataset.map(lambda ex: preprocess(ex))
        print("Dataset prorcessed...")

        # This aligns both the input_ids and the labels
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=self.model,
            padding="longest"
        ) 

        # This is to work with deepspeed and ZeRO stage 3
        output_dir = f"./Models/{save_dir}"
        training_args = TrainingArguments(
            output_dir = output_dir,
            learning_rate = 5e-5,
            num_train_epochs = epoch_num,
            save_steps = 500,
            logging_steps = 50,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            fp16=True,
            deepspeed="ds_config.json",
        )

        trainer = Trainer(
            model=self.model,
            args = training_args,
            train_dataset = processed_dataset,
            data_collator=data_collator
        )

        dl = trainer.get_train_dataloader()
        # item = next(iter(dl))
        # for key in item.keys():
        #     print(key, ": ", item[key].shape)

        # print(item["input_ids"])
        # print(item["labels"])

        # print("Start of training...")
        trainer.train()
        print("Training is finished, merging the outputs from GPU's")
        self._merge_finetune_outputs(output_dir)

    def _merge_finetune_outputs(self, output_dir):

        checkpoint_root = os.path.abspath(output_dir)

        for ckpt in sorted(os.listdir(checkpoint_root)):
            ckpt_path = os.path.join(checkpoint_root, ckpt)

            print("Merging: ", ckpt_path)
            subprocess.run(["python",
                os.path.join(ckpt_path, "zero_to_fp32.py"),
                ckpt_path,
                os.path.join(ckpt_path, "pytorch_model.bin"),])
 
            subprocess.run(f"mv {ckpt_path}/pytorch_model.bin/* {ckpt_path}/", shell=True)

    def _save_benchmark_results():
        # Needs the model and dataset to save in a .txt file 
        pass

    def load_dataset(self, dataset_name = "AI4Math/MathVista"):
        self.dataset_name = dataset_name
        dataset = load_dataset(dataset_name, split="testmini")
        self.dataset = dataset

    def benchmark(self, get_inputs, compare_outputs, dataset_name="AI4Math/MathVista"):
        
        self.load_dataset(dataset_name)
        self._prepare_for_inference()

        correct = 0
        total = 0
        for row in self.dataset:
            total+=1
            # Getting the image and the prompt from the dataset
            images, prompt = get_inputs(row)
            pred = self.run_inference(images=images, prompt=prompt).strip()
            correct+= compare_outputs(row, pred) 
            print("Accuracy: ", correct / total)

        return correct / total
