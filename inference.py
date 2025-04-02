import ast
import re
import pandas as pd
from tqdm import tqdm
import re

from CodeMMLU_RAG.model import tokenizer, model
from CodeMMLU_RAG.rag import RAG

# Read the questions
questions_df = pd.read_csv("./dataset/b6_test_data.csv")
questions_df["choices"] = questions_df["choices"].apply(ast.literal_eval)

def extract_index_from_output(text):
    # Search for the pattern after "<|im_start|>assistant"
    # match = re.search(r"<\|im_start\|>assistant\s*(?:ANSWER:\s*)?(\d+)", text)
    match = re.search(r"assistant\s*(?:ANSWER:\s*)?(\d+)", text)
    if match:
        return int(match.group(1))  # Extract the number
    else:
        return 0  # Default to 0 if no match is found

def format_prompt(question, choices, qa_pairs):
    formatted_choices = "\n".join([f"{i}. {c}" for i, c in enumerate(choices)])
    return f"""<|im_start|>system
You are a helpful AI assistant tasked with answering multiple-choice questions about coding.

I will provide you with questions and their possible answer choices. I will also give you several examples. For each question:
1. Respond solely with the index number of the correct option in the list, starting from 0.
2. Do not include any explanations, text, or anything other than the letter of the correct answer.

Here are some examples:

1. Question: {qa_pairs[0]['question']}
   Choices:
   {qa_pairs[0]['choices']}
   Answer: {qa_pairs[0]['answer']}
   Reasoning: {qa_pairs[0]['reasoning']}

2. Question: {qa_pairs[1]['question']}
   Choices:
   {qa_pairs[1]['choices']}
   Answer: {qa_pairs[1]['answer']}
   Reasoning: {qa_pairs[1]['reasoning']}

3. Question: {qa_pairs[2]['question']}
   Choices:
   {qa_pairs[2]['choices']}
   Answer: {qa_pairs[2]['answer']}
   Reasoning: {qa_pairs[1]['reasoning']}

Now, please answer the following question:
<|im_end|>
<|im_start|>user
Question: {question}
Choices:
{formatted_choices}
<|im_end|>
<|im_start|>assistant
"""

def extract_number_after_assistant(text):
    # Search for the number after "assistant" or "ANSWER:"
    match = re.search(r"assistant\s*(?:ANSWER:\s*)?(\d+)", text)
    if match:
        return int(match.group(1))  # Extract and return the number
    else:
        return None  # Return None if no match is found

def index_to_letter(index):
    try:
        letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        letter = letters[index]
    except:
        letter = 'A'
    return letter

def get_index_answer(prompt, max_new_tokens=5):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "<|im_start|>assistant" in decoded:
        result = decoded.split("<|im_start|>assistant")[-1].strip()
    else:
        result = decoded.strip()
    

    return index_to_letter((extract_index_from_output(result)))

preds = []

for idx, row in tqdm(questions_df.iterrows(), total=len(questions_df)):
    task_id = row['task_id']
    question = row['question']
    # print(question)
    choices_str = row['choices']

    qa_pairs = RAG.retrieval(idx, k=3)

    prompt = format_prompt(question, choices_str, qa_pairs)

    index = get_index_answer(prompt)
    preds.append(index)


submission = pd.DataFrame({
    "task_id": questions_df["task_id"],
    "answer": preds
})


submission.to_csv("submission.csv", index=False)
print("Saved to submission.csv.")