import gc
import os
import time
import warnings

import click
import pandas as pd
import torch
from openpyxl import Workbook
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline

def get_prompt(model_name):
    if model_name == 'microsoft/Phi-3-mini-4k-instruct' or model_name == 'microsoft/Phi-3-medium-4k-instruct':
        prompt = """<|user|>
Please identify the patient’s medical condition and current treatments, including any alternative names, abbreviations, or synonyms for these terms, as well as any additional criteria that may be important for identifying clinical trials of interest. Respond with a comma-separated list of keywords that will be used for search. Do not elaborate or explain. 

Patient’s medical note: {query}<|end|>
<|assistant|>"""
    elif model_name == 'Qwen/Qwen2-7B-Instruct':
        prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Please identify the patient’s medical condition and current treatments, including any alternative names, abbreviations, or synonyms for these terms, as well as any additional criteria that may be important for identifying clinical trials of interest. Respond with a comma-separated list of keywords that will be used for search. Do not elaborate or explain. 

Patient’s medical note: {query}<|im_end|>
<|im_start|>assistant
"""
    elif model_name == 'ruslanmv/Medical-Llama3-8B':
        prompt = """<|im_start|>system
You are an AI Medical Assistant trained on a vast dataset of health information. Respond with a comma-separated list of keywords. Do not elaborate or explain.<|im_end|>
<|im_start|>user
Patient’s medical note: {query}

What are the patient’s medical condition and current treatments, what are their alternative names, abbreviations, or synonyms, as well as tell if there are any additional criteria that may be important for identifying clinical trials of interest.<|im_end|>
<|im_start|>assistant
"""
    return prompt
def load_transformers_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        return tokenizer, model
    except ImportError as e:
        if "requires Accelerate" in str(e):
            print("Accelerate not found. Attempting to load model without device_map...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            return tokenizer, model
        else:
            raise e
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        print("Attempting to load with basic configuration...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        return tokenizer, model

def load_model(model_name):
    if model_name == 'UFNLP/gatortron-large':
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return {'tokenizer': tokenizer, 'model': model, 'config': config}
    else:
        return load_transformers_model(model_name)

@torch.no_grad()
def batch_infer(model_data, model_name, queries, batch_size=8):
    if isinstance(model_data, dict) and 'tokenizer' in model_data:
        # Gatortron-specific inference
        tokenizer, model = model_data['tokenizer'], model_data['model']
    else:
        tokenizer, model = model_data

    all_keywords = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i+batch_size]
        standard_prompt = get_prompt(model_name)
        full_prompts = [standard_prompt.format(query=query) for query in batch_queries]
        inputs = tokenizer(full_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.01, top_p=0.9, repetition_penalty=1.2)

        decoded_outputs = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        keywords = [output.strip() for output, prompt in zip(decoded_outputs, full_prompts)]
        all_keywords.extend(keywords)

    return all_keywords

def process_dataframe(df, output_folder, model_data, model_name):
    print(f"\nProcessing {df.name} with {model_name}")
    print(f"Saving results to: {output_folder}")

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{df.name}_{model_name.replace('/', '_')}.xlsx")

    wb = Workbook()
    ws = wb.active
    ws.append(["qid", "query", "generated_keywords", "processing_time"])

    total_queries = len(df)
    total_time = 0
    batch_size = 1  # Adjust based on your GPU memory

    for i in tqdm(range(0, total_queries, batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        start_time = time.time()

        try:
            batch_keywords = batch_infer(model_data, model_name, batch_df['query'].tolist(), batch_size)

            for (_, row), keywords in zip(batch_df.iterrows(), batch_keywords):
                ws.append([row['qid'], row['query'], keywords, None])  # We'll update processing time later

            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time

            # Update processing time for each query in the batch
            for j in range(batch_size):
                if i + j < total_queries:
                    ws.cell(row=i+j+2, column=4, value=batch_time / len(batch_df))

        except Exception as e:
            print(f"Error processing batch: {e}")
            for _, row in batch_df.iterrows():
                ws.append([row['qid'], row['query'], "Error generating keywords", None])

        if (i + batch_size) % (batch_size * 10) == 0:
            wb.save(output_file)
            print(f"Progress saved. Completed {i + batch_size}/{total_queries} queries.")

    wb.save(output_file)
    print(f"\nCompleted processing {df.name} with {model_name}.")
    print(f"Results saved to: {output_file}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per query: {total_time/total_queries:.2f} seconds")
