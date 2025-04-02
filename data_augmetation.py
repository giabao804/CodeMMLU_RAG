
import pandas as pd
import torch
from tqdm import tqdm
from CodeMMLU_RAG.data_preprocessing import char_to_index, extract_answer_char
from CodeMMLU_RAG.model import tokenizer, model



class DataAugmentor:
    def __init__(self, data_path, batch_size=8, max_new_tokens=256):
        self.data_path = data_path
        self.data = pd.read_csv(data_path)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

    def format_prompt(question, answer):
        return f"""<|im_start|>system
            Given a question and its correct answer. Explain why this is the correct answer. Be concise and clear in your explanation.
            <|im_end|>
            <|im_start|>user
            Explain why this is the correct answer:
            {question}
            Correct Answer:
            {answer}
            <|im_end|>
            <|im_start|>assistant
            """    

    def generate_reasoning(self):
        reasonings = {}

        # Use a batch processing approach
        num_batches = len(self.data) // self.batch_size + (1 if len(self.data) % self.batch_size > 0 else 0)

        for batch_idx in tqdm(range(num_batches), total=num_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min((batch_idx + 1) * self.batch_size, len(self.data))
            batch_df = self.data.iloc[batch_start:batch_end]

            # Prepare the inputs for the batch
            batch_prompts = []
            task_ids = []

            for _, row in batch_df.iterrows():
                task_id = row['task_id']
                question = row['question']
                choices_str = row['choices']
                answer_raw = row['answer']

                # Extract clean answer char
                answer_char = extract_answer_char(answer_raw)
                if not answer_char:
                    continue  # Skip rows with no valid answer

                try:
                    choices = eval(choices_str)
                except Exception:
                    continue  # Skip rows where choices can't be evaluated

                answer_index = char_to_index(answer_char)

                # Get correct answer text
                if 0 <= answer_index < len(choices):
                    correct_answer_text = choices[answer_index]

                prompt = self.format_prompt(question, correct_answer_text)
                batch_prompts.append(prompt)
                task_ids.append(task_id)

            # Tokenize the entire batch and pass to the model
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

            # Generate outputs for the entire batch in one go
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                top_p=1.0,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # Decode the outputs for the entire batch
            for task_id, output in zip(task_ids, outputs):
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                if "<|im_start|>assistant" in decoded:
                    reasoning = decoded.split("<|im_start|>assistant")[-1].strip()
                else:
                    reasoning = decoded.strip()

                # Save each task_id with the corresponding reasoning
                reasonings[task_id] = reasoning

            # Clear cache and free memory after each batch
            del inputs  # Delete the input batch
            del outputs  # Delete the model's output batch
            torch.cuda.empty_cache()  # Clear the GPU memory cache

        return reasonings

    def save_to_csv(self, output_path):
        # Generate reasonings
        reasonings = self.generate_reasoning()

        # Add the reasonings to a new column in the DataFrame
        self.data['reasoning'] = self.data['task_id'].map(reasonings)

        # Save the updated DataFrame to a new CSV file
        self.data.to_csv(output_path, index=False)
        print(f"Data with reasonings saved to {output_path}")