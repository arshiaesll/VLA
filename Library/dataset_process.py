
from datasets import load_dataset
# from model_manager import ModelManager
print("Libraries loaded!")

class AdapterProvider():
     
    def __init__(self, dataset_name) -> None:

        # These are the simplified versions  
        if dataset_name == "mathvista":
            self.get_inputs = math_vista_get_inputs
            self.compare_outputs = math_vista_compare_outputs
            self.preprocess = math_vista_preprocess
            self.split = "testmini"

        elif dataset_name == "mathdial" :
            self.preprocess = math_dial_preprocess_v3
            self.split = "train"


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

def math_dial_preprocess(example, processor):
    conv = example["conversation"]
    lst = conv.split("|EOM|")

    students = []
    teachers = []

    for idx, item in enumerate(lst):
        if idx % 2 == 0:
            teachers.append(item)
        else:
            students.append(item)

    messages = []
    for teacher, student in zip(teachers, students):
        messages.append({"role": "user", "content": student[9:]})
        messages.append({"role": "assistant", "content": teacher[9:]})

    full_prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


    model_inputs = processor(
        text=full_prompt
    )

    input_ids = model_inputs["input_ids"]
    input_ids[input_ids == -1] = processor.tokenizer.pad_token_id
    model_inputs["input_ids"] = input_ids

    model_inputs = {k: v[0] for k, v in model_inputs.items()}
    return model_inputs

def math_dial_preprocess_v2(example, processor):

    conv = example["conversation"]
    lst = conv.split("|EOM|")
    user_token_id = processor.tokenizer.encode("<|user|>")[1]
    end_token_id = processor.tokenizer.encode("<|end|>")[1]

    students = []
    teachers = []

    for idx, item in enumerate(lst):
        if idx % 2 == 0:
            teachers.append(item)
        else:
            students.append(item)

    messages = []
    for teacher, student in zip(teachers, students):
        messages.append({"role": "user", "content": student[9:]})
        messages.append({"role": "assistant", "content": teacher[9:]})

    full_prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


    model_inputs = processor(
        text=full_prompt
    )

    input_ids = model_inputs["input_ids"]
    input_ids[input_ids == -1] = processor.tokenizer.pad_token_id
    # model_inputs["input_ids"] = input_ids
    # model_inputs["labels"] = model_inputs["input_ids"].clone()



    model_inputs["input_ids"] = input_ids[0].tolist()
    model_inputs["labels"] = input_ids[0].tolist()
    model_inputs["attention_mask"] = model_inputs["attention_mask"][0]
    # model_inputs = {k: v[0] for k, v in model_inputs.items()}

    turn_to_neg = False
    for idx, val in enumerate(model_inputs["labels"]):
      if val == user_token_id:
        turn_to_neg = True
        continue 
      elif val == end_token_id:
        turn_to_neg = False

      if turn_to_neg:
        model_inputs["labels"][idx] = -100
    return model_inputs


def math_dial_preprocess_v3(example, processor):

    user_token_id = processor.tokenizer.encode("<|user|>")[1]
    end_token_id = processor.tokenizer.encode("<|end|>")[1]
    assistant_token_id = processor.tokenizer.encode("<|assistant|>")[1]

    context_text = f"""
    Here is the context for this problem:
    
    Student profile:
    {example['student_profile']}
    
    Question:
    {example['question']}
    
    Correct answer:
    {example['ground_truth']}
    
    Student’s incorrect answer:
    {example['student_incorrect_solution']}
    
    Now continue with the following conversation:
    """

    messages = [{"role": "user", "content": context_text}]

    conv = example["conversation"]
    lst = conv.split("|EOM|")
    students = []
    teachers = []

    for idx, item in enumerate(lst):
        if idx % 2 == 0:
            teachers.append(item)
        else:
            students.append(item)

    for teacher, student in zip(teachers, students):
        messages.append({"role": "assistant", "content": teacher[9:]})
        messages.append({"role": "user", "content": student[9:]})

    if len(teachers) > len(students):
      messages.append({"role": "assistant", "content": teachers[-1][9:]})

    full_prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


    model_inputs = processor(
        text=full_prompt
    )
    input_ids = model_inputs["input_ids"]
    input_ids[input_ids == -1] = processor.tokenizer.pad_token_id
    # model_inputs["input_ids"] = input_ids
    # model_inputs["labels"] = model_inputs["input_ids"].clone()



    model_inputs["input_ids"] = input_ids[0].tolist()
    model_inputs["labels"] = input_ids[0].tolist()
    model_inputs["attention_mask"] = model_inputs["attention_mask"][0]
    # model_inputs = {k: v[0] for k, v in model_inputs.items()}

    turn_to_neg = True
    for idx, val in enumerate(model_inputs["labels"]):
      if val == assistant_token_id:
        turn_to_neg = False
      elif val == end_token_id:
        turn_to_neg = True
        continue

      if turn_to_neg:
        model_inputs["labels"][idx] = -100
    return model_inputs





if __name__ == "__main__":
    ds = load_dataset("eth-nlped/mathdial", split="train")
    manager = ModelManager()
    manager.fine_tune(dataset_name="eth-nlped/mathdial", split="train", preprocess = lambda ex: math_dial_preprocess_v2(ex, manager.processor)) 

    # manager.fine_tune(dataset_name="AI4Math/MathVista", split="testmini", preprocess = lambda ex: math_vista_preprocess(ex, manager.processor)) 