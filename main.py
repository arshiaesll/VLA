from Library.model_manager import ModelManager


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

if __name__ == "__main__":
    manager = ModelManager(fine_tuned_path="./Models/phi-3.5_MathVista/checkpoint-1000")
    manager.benchmark(get_inputs = math_vista_get_inputs, 
                      compare_outputs= math_vista_compare_outputs,
                      dataset_name= "AI4Math/MathVista")
    