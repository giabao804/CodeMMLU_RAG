import numpy as np
import pandas as pd

# Function to convert A/B/C... to index
def char_to_index(char):
    return ord(char) - ord('A')

# Function to extract character (A, B, C...) from answer field
def extract_answer_char(answer):
    if pd.isna(answer) or not isinstance(answer, str):
        return None
    if "ANSWER:" in answer:
        answer_char = answer.split(":")[-1].strip()
    else:
        answer_char = answer.strip()
    return answer_char

def data_preprocessing(data_path):
    # Load the data
    data = pd.read_csv(data_path)

    