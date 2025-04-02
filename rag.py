import pickle
import torch
from torch.nn import functional as F
import numpy as np
import pandas as pd


class RAG:
    def __init__(self, train_csv_data, train_pkl_path, test_pkl_path):
        self.train_csv_data = train_csv_data
        self.train_pkl_path = train_pkl_path
        self.test_pkl_path = test_pkl_path

    @staticmethod
    def vector_store(file_path):
        data = pickle.load(open(file_path, "rb"))

        task_id = list(data.keys())
        embeddings = np.array(list(data.values()))
        embeddings = torch.tensor(embeddings)

        matrix_embeddings = torch.cat([embeddings], dim=0)
        normalized_embeddings = F.normalize(matrix_embeddings, p=2, dim=1)

        return task_id, normalized_embeddings

    @staticmethod
    def top_k_cosine_similarity(A, B, k=3):
        
        A_norm = A / A.norm(dim=1, keepdim=True)
        B_norm = B / B.norm(dim=1, keepdim=True)

        similarity = torch.mm(A_norm, B_norm.T)  # [n, m] matrix

        top_k_values, top_k_indices = torch.topk(similarity, k=k, dim=1, largest=True, sorted=True)

        return top_k_values, top_k_indices

    def retrieval(self, query_index, k=3):
        # Load vector stores for train and test data
        task_id, train_db = self.vector_store(self.train_pkl_path)
        _, test_db = self.vector_store(self.test_pkl_path)

        # Retrieve top-k most similar questions
        top_k_values, top_k_indices = self.top_k_cosine_similarity(test_db, train_db, k=k)
        rag_indices = top_k_indices[query_index].tolist()
        top_k_task_id = [task_id[idx] for idx in rag_indices]

        # Load raw data and filter by top-k task IDs
        raw_data = pd.read_csv(self.train_csv_data)
        qa_pairs = raw_data[raw_data["task_id"].isin(top_k_task_id)][["question", "choices", "answer", "reasoning"]]

        return qa_pairs.to_dict(orient='records')


