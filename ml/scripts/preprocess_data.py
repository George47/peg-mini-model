import pandas as pd
import numpy as np

def preprocess_data(input_file, output_file):
    df = pd.read_csv(input_file)
    # Implement your preprocessing logic here
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data('data/tasks.csv', 'data/preprocessed_tasks.csv')