from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig, Trainer, TrainingArguments, DataCollatorWithPadding, DataCollatorForSeq2Seq
import torch
# from PIL import Image
import flash_attn
import requests
from datasets import load_dataset
import os

print("Libraries Loaded")
print("CUDA Available: ", torch.cuda.is_available())
print("CUDA Device count: ", torch.cuda.device_count())
print("World size:", os.environ.get("WORLD_SIZE"))
print("Rank:", os.environ.get("RANK"))
print("Local rank:", os.environ.get("LOCAL_RANK"))

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: ", torch.cuda.get_device_name(i))


class ModelManager():

    def __init__(self, save_dir, model_name = "microsoft/phi-3.5-vision-instruct"):
        self.save_dir = save_dir
        self.model_path = model_name
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code = True)

        self.config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
            
        # Using normal attention
        self.config._attn_implementation = "eager"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config = self.config,
            torch_dtype = torch.float16,
            trust_remote_code = True,
            )

        # self.model.config.use_cache = False 
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

    def run_inference(self, image, prompt):

        images = []
        messages = messages = [
            {"role": "user", "content": prompt + "<|image_1|>\n"}
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        images = [image]

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

    def gpt_decoding_func(self, input_ids, labels, tokenizer):
        """
        Debug function to decode and compare input_ids vs labels.
        - input_ids: torch.Tensor or list of token IDs
        - labels: torch.Tensor or list of token IDs (-100 masked)
        - tokenizer: your Hugging Face tokenizer
        """

        # Convert tensors to lists if needed
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if hasattr(labels, "tolist"):
            labels = labels.tolist()

        # If batched, take first row
        if isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if isinstance(labels[0], list):
            labels = labels[0]

        # Decode input_ids normally
        decoded_input = tokenizer.decode([i for i in input_ids if i >= 0])

        # Decode labels but skip -100 (masked)
        decoded_labels = tokenizer.decode([l for l in labels if l != -100 and l >= 0])

        print("Input IDs (full conversation):")
        print(decoded_input)
        print("="*80)
        print("Labels (only assistant answer):")
        print(decoded_labels)
        print("="*80)

        # Quick stats
        print("Input length:", len(input_ids))
        print("Label length:", len(labels))
        print("Masked tokens in labels:", sum(1 for l in labels if l == -100))

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

        # print(full_prompt)

        model_inputs = self.processor(
            text=full_prompt,
            images=image,
            # truncation=True,
            # padding="max_length",
            # max_length = 512
        )

        # user_input_ids = self.processor.tokenizer.apply_chat_template(
        #     messages[:-1],
        #     tokenize = True,
        #     add_generation_prompt = True
        # )

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

    def fine_tune(self, dataset_name):
        # Loading the dataset
        self.load_dataset(dataset_name)
        processed_dataset = self.dataset.map(lambda ex: self.preprocess(ex))
        print("Dataset prorcessed...")

        # This aligns both the input_ids and the labels
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.processor.tokenizer,
            model=self.model,
            padding="longest"
        ) 

        # This is to work with deepspeed and ZeRO stage 3
        training_args = TrainingArguments(
            output_dir = self.save_dir,
            learning_rate = 5e-5,
            num_train_epochs = 2,
            save_steps = 200,
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
        item = next(iter(dl))
        for key in item.keys():
            print(key, ": ", item[key].shape)

        print(item["input_ids"])
        print(item["labels"])

        print("Start of training...")
        trainer.train()

    def load_dataset(self, dataset_name = "AI4Math/MathVista"):
        self.dataset_name = dataset_name
        dataset = load_dataset(dataset_name, split="testmini")
        self.dataset = dataset

    def make_prompt(self, choices, question):
        return f"Answer the following question: {question}, pick from these chioces: {choices}"

    def benchmark(self, dataset_name):
        
        self.load_dataset(dataset_name)
        self._prepare_for_inference()

        correct = 0
        total = 0
        for row in self.dataset:
            total+=1
            # Getting the image and the prompt from the dataset
            if dataset_name == "AI4Math/MathVista":
                image = row["decoded_image"]
                prompt = self.make_prompt(row["choices"], row["question"])
                pred = self.run_inference(image=image, prompt=prompt).strip()

                answer = row["answer"].strip()

                if answer.lower() in pred.lower():
                    correct+=1
                
            print("Accuracy: ", correct / total)
        return correct / total


if __name__ == "__main__":

    manager = ModelManager(save_dir="./phi-3.5_MathVista_with_image")
    # image = Image.open("./test2.jpg").convert("RGB")
    # print(manager.run_inference(image, prompt="Describe the image"))
    manager.benchmark(dataset_name= "AI4Math/MathVista")
    # manager.fine_tune(dataset_name = "AI4Math/MathVista") 
    # manager.benchmark()