from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
import os
import subprocess

print("Libraries Loaded")
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device count: ", torch.cuda.device_count())
print("World size:", os.environ.get("WORLD_SIZE"))
print("Rank:", os.environ.get("RANK"))
print("Local rank:", os.environ.get("LOCAL_RANK"))

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: ", torch.cuda.get_device_name(i))


class ModelManager():

    def __init__(self, base_model="microsoft/Phi-3.5-vision-instruct", fine_tuned_path = ""):
        
        # base_model = "microsoft/phi-3-mini-4k-instruct"
        self.base_model = base_model
        self.processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code = True)

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model if fine_tuned_path == "" else fine_tuned_path,
            # Using flash attention
            # The new version has dtype instead of torch_dtype
            torch_dtype = torch.float16,
            trust_remote_code = True,
            )
        self.processor.tokenizer.eos_token = "<|end|>"
        self.processor.tokenizer.pad_token = "<|end|>"
        self.model.config.eos_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|end|>")
        self.model.config.pad_token_id = self.model.config.eos_token_id

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
        print(f"Using {self.device}")
        self.model.to(self.device)

    # images is a list of image, prompt is str
    def run_inference(self, prompt = "", images = [], messages = []):
        
        image_str_lst = [f"<|image_{i}|>" for i in range(len(images))]
        image_str = "\n".join(image_str_lst)

        if messages:
            pass
        else:
            messages = messages = [
                {"role": "user", "content": prompt + image_str}
            ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            prompt,
            images if len(images) > 0 else None
        ).to(self.device)

        with torch.inference_mode():
            # Using KV Cache for generation
            output = self.model.generate(
                **inputs,
                max_new_tokens = 200,
                eos_token_id = self.model.config.eos_token_id,
                pad_token_id = self.model.config.eos_token_id,
            )

        output = output[:, inputs["input_ids"].shape[1] :]
        clean_ids = output.masked_fill(output == -1, self.processor.tokenizer.eos_token_id)
        response = self.processor.tokenizer.decode(clean_ids[0], skip_special_tokens = True)

        return response

    # Outputs in base_model-dataset_name-uuid
    def fine_tune(self, dataset_name:str, split:str, preprocess, save_path , epoch_num=2):
        # Loading the dataset
        self.load_dataset(dataset_name, split)
        # self.freeze_LLM_part()

        print("Model: ", self.base_model)
        print("Dataset: ", self.dataset_name)
        print("Epochs: ", epoch_num)

        # save_dir = f"{self.base_model}-{self.dataset_name}"
        processed_dataset = self.dataset.map(lambda ex: preprocess(ex), remove_columns = self.dataset.column_names)
        print("Dataset prorcessed...")
        # This aligns both the input_ids and the labels
        print(processed_dataset)
        print(processed_dataset["input_ids"][0])
        print(processed_dataset["labels"][0])
        print(processed_dataset["attention_mask"][0])

        self.processor.tokenizer.padding_side = "left"
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=self.model,
            padding="longest"
        ) 
 
        sample = [processed_dataset[0], processed_dataset[1]]   # pick two rows
        batch = data_collator(sample)
        # print(batch["input_ids"].shape, batch["labels"].shape)
        
        # This is to work with accelerate launch ...
        # output_dir = f"./Models/{save_dir}"
        training_args = TrainingArguments(
            output_dir = save_path,
            learning_rate = 1e-5,
            num_train_epochs = epoch_num,
            warmup_steps=50,
            bf16=True,
            save_steps = 1500,
            logging_steps = 100,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_grad_norm = 1.0
        )

        trainer = Trainer(
            model=self.model,
            args = training_args,
            train_dataset = processed_dataset,
            data_collator=data_collator
        )

        # batch = next(iter(trainer.get_train_dataloader()))
        # print("input_ids: ", batch["input_ids"][0])
        # print("labels:", batch["labels"][0][:50])
        # print("valid tokens:", (batch["labels"] != -100).sum().item())

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

    def load_dataset(self, dataset_name = "AI4Math/MathVista", split="testmini"):
        self.dataset_name = dataset_name
        dataset = load_dataset(dataset_name, split=split)
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


