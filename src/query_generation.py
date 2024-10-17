import gc
import os
import time
import warnings

import click
import pandas as pd
import torch
from bs4 import BeautifulSoup
from openpyxl import Workbook
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline

from utils import load_model, batch_infer, process_dataframe

def get_trec_datasets(dataset_folder):
    with open(os.path.join(dataset_folder, 'trec_cds/topics-2014_2015-description.topics'), 'r') as document:
        d = document.read()
        soup = BeautifulSoup(d, 'xml')
        qid = soup.find_all('NUM')
        query = soup.find_all('TITLE')
        lq = []
        for i in qid:
            lq.append(i.text)
        ld = []
        for x in query:
            ld.append(x.text)
    trec_cds = pd.DataFrame({'qid': lq,'query': ld})

    trec_cds["query"] = trec_cds["query"]
    trec_cds["qid"] = trec_cds["qid"].apply(lambda x: x.replace('2014','').replace('2015',''))
    trec_cds["qid"] = trec_cds["qid"].astype(str)

    path_q21 = os.path.join(dataset_folder, 'trec_21/queries_2021.xlsx')
    path_q22 = os.path.join(dataset_folder, 'trec_22/queries_2022.xlsx')
    trec_21 = pd.read_excel(path_q21 ,names = ['qid','query'], converters={'qid':str},index_col= False)
    trec_22 = pd.read_excel(path_q22 ,names = ['qid','query'], converters={'qid':str,'query':str},index_col= False)

    return trec_cds, trec_21, trec_22

@click.command()
@click.option('--model_name', prompt='Model Name', help='The name of the model to use.')
@click.option('--dataset_folder', prompt='Dataset Folder', help='The folder where the dataset is located.')
def main(model_name, dataset_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trec_cds, trec_21, trec_22 = get_trec_datasets(dataset_folder)
    # Load dataframes and models
    dataframes = [
        ('trec_21',  trec_21),
        ('trec_22', trec_22),
        ('trec_cds', trec_cds)
    ]

    models = [
        'microsoft/Phi-3-mini-4k-instruct',
        'microsoft/Phi-3-medium-4k-instruct',
        'Qwen/Qwen2-7B-Instruct',
        'ruslanmv/Medical-Llama3-8B'
    ]
    assert model_name in models, f"Model {model_name} not found in available models: {models}"

    print(f"\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}")

    try:
        print(f"Loading model: {model_name}")
        model_data = load_model(model_name)
        print(f"Model {model_name} loaded successfully")

        for df_name, df in dataframes:
            if df is None:
                print(f"Skipping dataframe {df_name} due to loading error")
                continue
            df.name = df_name
            output_folder = os.path.join(dataset_folder, df_name) + '/'
            process_dataframe(df, output_folder, model_data, model_name)

        print(f"Completed processing all dataframes with {model_name}")

    except Exception as e:
        print(f"Error processing with {model_name}: {str(e)}")
        print("Traceback:")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup, even if an error occurred
        if 'model_data' in locals():
            del model_data
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Cleanup completed for {model_name}")

    print(f"{'='*50}\n")

    print("All dataframes processed.")


if __name__ == '__main__':
    main()