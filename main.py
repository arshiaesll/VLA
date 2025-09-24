from Library.model_manager import ModelManager
from Library.dataset_process import AdapterProvider
import argparse




if __name__ == "__main__":

    # This is to make datasetnames easier to pass as arguments
    dataset_dict = {
        "mathvista": "AI4Math/MathVista",
        "mathdial": "eth-nlped/mathdial"
    }

    parser = argparse.ArgumentParser() 
    parser.add_argument("mode", choices=["train", "benchmark", "inference"])
    parser.add_argument("--train-dataset", type=str, default="mathvista", help="Supported datasets are: mathvista, mathdial")
    parser.add_argument("--benchmark-dataset", type=str, default="mathvista", help="Supported datasets are: mathvista, mathdial")
    parser.add_argument("--from-finetuned", type=str, default=None, help="The path to the checkpoint of the fine-tuned model, ex:./Models/phi-3.5_MathVista/checkpoint-1000")
    parser.add_argument("--output-dir", type=str, default="./Models", help="The output directory for fine-tuned models")
    parser.add_argument("--run-name", type=str, help="Name your run(this will be used in the saved place s.t ./Model/{run_name})")
    parser.add_argument("--inference-dataset", default="mathdial", type=str, help="The dataset to test the inference on")

    args = parser.parse_args()
    if args.run_name:
        full_fine_tune_path = args.output_dir + "/" + args.run_name
    else:
        full_fine_tune_path = args.output_dir

    if args.mode == "train" and (not args.train_dataset or not args.run_name):
        parser.error("--train-dataset is required when mode is 'train'")

    if args.mode == "benchmark" and not args.benchmark_dataset:
        parser.error("--benchmark-dataset is required when mode is 'benchmark'")

    if args.from_finetuned:
         print("Loading model from: ", args.from_finetuned)
         manager = ModelManager(fine_tuned_path=args.from_finetuned)
    else:
         manager: ModelManager = ModelManager()

    if args.mode == "train":
        print("Preparing the training!") 
        dataset_name = args.train_dataset 
        provider = AdapterProvider(dataset_name=dataset_name)
        split = provider.split
        manager.fine_tune(dataset_dict[dataset_name], split=split, preprocess= lambda ex: provider.get_preprocess()(ex, manager.processor), save_path = full_fine_tune_path)

    elif args.mode == "benchmark":
        print("Preparing Benchmark!")
        dataset_name = args.benchmark_dataset
        provider = AdapterProvider(dataset_name=dataset_name)
        manager.benchmark(provider.get_inputs, provider.compare_outputs, dataset_name=dataset_dict[dataset_name])

    elif args.mode == "inference":
        print("Setting up the inference mode, put 'q' to quit")
        user = ""
        manager._prepare_for_inference()
        dataset_name = args.inference_dataset
        args.inference_dataset
        manager.load_dataset(dataset_name = dataset_dict[dataset_name], split="test")
        example = manager.dataset[1]
        context = f"""
            Here is the context for this problem:
    
            Student profile:
            {example['student_profile']}
            
            Question:
            {example['question']}
            
            Correct answer:
            {example['ground_truth']}
             
        """
        messages = []
        messages.append({"role": "user", "content": context})
        user = input(f"Type the answer to this question: {example['question']}")
        messages.append({"role": "user", "content": user})
        while user != "q":
            response = manager.run_inference(messages = messages)
            print("Model: ", response + "\n")
            messages.append({"role": "assistant", "content": response})
            user = input()
            messages.append({"role": "user", "content": user})

    # manager = ModelManager(fine_tuned_path="./Models/phi-3.5_MathVista/checkpoint-1000")
    # manager.benchmark(get_inputs = math_vista_get_inputs, 
    #                   compare_outputs= math_vista_compare_outputs,
    #                   dataset_name= "AI4Math/MathVista")
    # manager._merge_finetune_outputs("./Models/phi-3.5_MathVista_with_image")
    # dataset_name = "AI4Math/MathVista"
    # manager.fine_tune(dataset_name=dataset_name, preprocess = lambda ex: math_vista_preprocess(ex, manager.processor))
    