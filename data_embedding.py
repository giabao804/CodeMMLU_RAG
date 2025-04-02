import torch.nn.functional as F
from torch import Tensor
import os
import pandas as pd
import pickle
import gc
from tqdm import tqdm
from CodeMMLU_RAG.model import embedding_model, embedding_tokenizer

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embedding_batch(texts, device):
   
    inputs = embedding_tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
    outputs = embedding_model(**inputs)
    embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
    
    return embeddings

device = "cpu"
embedding_model.to(device)

def convert_pkl(file_path, output_path, file_name):
    batch_size = 8
    embeddings_list = []

    chunk_size = 100  # Read in chunks to avoid OOM
    for idx, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        # Preprocess chunk
        chunk["question"] = chunk["question"].str.replace("^Question: ", "", regex=True)

        
        # Process each batch in the chunk
        for start_idx in tqdm(range(0, len(chunk), batch_size), desc="Processing Batches"):
            batch_texts = chunk["question"].iloc[start_idx:start_idx + batch_size].to_list()
            embeddings = embedding_batch(batch_texts, device).detach().numpy()
            embeddings_list.extend(embeddings)

        # After processing each chunk, clear out chunk-related variables
        chunk["embeddings"] = embeddings_list
        embeddings_list.clear()  # Clear embeddings_list to free memory

        # Create task dictionary from chunk
        task_dict = pd.Series(chunk["embeddings"].values, index=chunk["task_id"]).to_dict()
        task_dict = {k: v for k, v in task_dict.items() if v is not None}

        # Save the task dictionary to pickle
        with open(f"{output_path}_v{idx}.pkl", "wb") as f:
            pickle.dump(task_dict, f)

        print(f"saved {output_path}_v{idx}.pkl")
        
        # Free memory by deleting chunk and invoking garbage collection
        del chunk
        gc.collect()

    task_dict = {}
    for file in os.listdir(output_path):
        if file.endswith(".pkl"):
            data = pickle.load(open(os.path.join(output_path, file), "rb"))
            task_dict.update(data)

    with open(file_name, "wb") as f:
        pickle.dump(task_dict, f)  


    return None      


file_path = "./dataset/train_data.csv"
output_path = "train_data"
file_name = "train_data.pkl"
convert_pkl(file_path, output_path, file_name)