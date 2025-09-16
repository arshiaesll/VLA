from Library.model_manager import ModelManager
import argparse


class AdapterProvider():
     
    def __init__(self, dataset_name) -> None:
        
        if dataset_name == "AI4Math/MathVista":
            self.get_inputs = math_vista_get_inputs
            self.compare_outputs = math_vista_compare_outputs
            self.preprocess = math_vista_preprocess
    
    def get_get_inputs(self):
        return self.get_get_inputs 

    def get_compare_outputs(self):
        return self.compare_outputs

    def get_preprocess(self):
        return self.preprocess





def math_vista_make_prompt(choices, question):
    return f"Answer the following question: {question}, pick from these chioces: {choices}"

def math_vista_get_inputs(row):
    image = row["decoded_image"]
    prompt = math_vista_make_prompt(row["choices"], row["question"])
    return [image], prompt

def math_vista_compare_outputs(row, pred):
    if row["answer"].lower() in pred.lower():
        return 1
    else:
        return 0

def math_vista_preprocess(example, processor):
        image = example["decoded_image"]
        messages = [
            {"role": "user", "content": f"{example['question']} {example['choices']}" + "<|image_1|>\n"},
            {"role": "assistant", "content": example["answer"]}
        ]

        full_prompt = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        model_inputs = processor(
            text=full_prompt,
            images=image,
        )

        input_ids = model_inputs["input_ids"]
        input_ids[input_ids == -1] = processor.tokenizer.pad_token_id
        model_inputs["input_ids"] = input_ids
        labels = model_inputs["input_ids"].clone()
 
        model_inputs["labels"] = labels
        # for key in model_inputs.keys():
        #     print(key, ": ", model_inputs[key].shape)

        model_inputs = {k: v[0] for k, v in model_inputs.items()}
        # print("Input ids: ", model_inputs["input_ids"])
        # print("Labels: ", model_inputs["labels"])

        return model_inputs 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("mode", choices=["train", "benchmark"])
    parser.add_argument("--train-dataset", type=str, default="AI4Math/MathVista", help="Supported datasets are: AI4Math/MathVista")
    parser.add_argument("--benchmark-dataset", type=str, default="AI4Math/MathVista", help="Supported datasets are: AI4Math/MathVista")
    parser.add_argument("--from-finetuned", type=str,default=None, help="The path to the checkpoint of the fine-tuned model, ex:./Models/phi-3.5_MathVista/checkpoint-1000")

    args = parser.parse_args()

    if args.mode == "train" and not args.train_dataset:
        parser.error("--train-dataset is required when mode is 'train'")

    if args.mode == "benchmark" and not args.benchmark_dataset:
        parser.error("--benchmark-dataset is required when mode is 'benchmark'")

    if args.from_finetuned:
         print("Loading model from: ", args.from_finetuned)
         manager = ModelManager(fine_tuned_path=args.from_finetuned)
    else:
         manager = ModelManager()

    if args.mode == "train":
         dataset_name = args.train_dataset 
         provider = AdapterProvider(dataset_name=dataset_name)
         manager.fine_tune(dataset_name, preprocess= lambda ex: provider.get_preprocess()(ex, manager.processor))

    elif args.mode == "benchmark":
        dataset_name = args.benchmark_dataset
        provider = AdapterProvider(dataset_name=dataset_name)
        manager.benchmark(provider.get_get_inputs, provider.get_compare_outputs, dataset_name=dataset_name)


    # manager = ModelManager(fine_tuned_path="./Models/phi-3.5_MathVista/checkpoint-1000")
    # manager.benchmark(get_inputs = math_vista_get_inputs, 
    #                   compare_outputs= math_vista_compare_outputs,
    #                   dataset_name= "AI4Math/MathVista")
    # manager._merge_finetune_outputs("./Models/phi-3.5_MathVista_with_image")
    # dataset_name = "AI4Math/MathVista"
    # manager.fine_tune(dataset_name=dataset_name, preprocess = lambda ex: math_vista_preprocess(ex, manager.processor))
    