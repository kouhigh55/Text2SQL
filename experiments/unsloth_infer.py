# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Infer unsloth model
# Note:

import argparse
import os
import re
import sys
import json

import pandas as pd
from datasets import Dataset
from torch.utils.data import DataLoader

sys.path.append(os.path.split(sys.path[0])[0])

from unsloth import FastLanguageModel, get_chat_template
from tqdm import tqdm

prompt_base = """
Turn this natural language query into standard SQL language.
query: {ques},
table_schema: {sche},
SQL result:
"""


def construct_prompt(question, schema):
    return prompt_base.format(ques=question, sche=schema)


class SpiderDataset(Dataset):
    def __init__(self, data_frame, schema):
        self.data_frame = data_frame
        self.schema = schema

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        sch = self.schema[row['db_id']]
        sentence = row['question']
        return construct_prompt(sentence, sch)


def collate_fn(batch, tokenizer):
    # Prepare messages for input
    messages = [{"role": "user", "content": sentence} for sentence in batch]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # Must add for generation
        return_tensors="pt",
    )
    return inputs


def infer(args):
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    # Load Spider schema
    with open(args.schema, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    schema_dict = {entry["db_id"]: entry["Schema (values (type))"] for entry in json_data}

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    # Load Spider dataset
    splits = {'train': 'spider/train-00000-of-00001.parquet', 'validation': 'spider/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/xlangai/spider/" + splits["validation"])

    dataset = SpiderDataset(df, schema_dict)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    predicted_ls = []

    for inputs in tqdm(dataloader, desc="Processing Batches"):
        inputs = inputs.to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_seq_length,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

        inference_outputs = tokenizer.batch_decode(outputs)

        for inference_output in inference_outputs:
            output_text = re.search(
                r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
                inference_output,
                re.DOTALL,
            )

            if output_text:
                real_output = output_text.group(1).strip()
            else:
                real_output = ""
                print(f"Error when parsing real output from {inference_output}")

            predicted_ls.append(real_output)

    # Save the outputs
    with open(args.save_dir, "w") as file:
        for line in predicted_ls:
            file.write(line + "\n")

    if args.show_infer:
        for input_sentence, output_sentence in zip(dataset, predicted_ls):
            print(f"Input: {input_sentence}\nGenerated: {output_sentence}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_name',
        '-m',
        type=str,
        required=True,
        help='Path or name to fine-tuned model',
    )

    parser.add_argument(
        '--save_dir',
        '-sd',
        type=str,
        required=True,
        help='Path or name to fine-tuned model',
    )

    parser.add_argument(
        '--schema',
        '-sc',
        type=str,
        required=False,
        default='./home/data/schema.json',
        help='Path or name to pre-trained model',
    )

    parser.add_argument('--max_seq_length', type=int, required=False, default=512)
    parser.add_argument('--load_in_4bit', type=bool, required=False, default=True,
                        help='Use 4bit quantization to reduce memory usage. Can be False')
    parser.add_argument('--show_infer', type=bool, required=False, default=False)

    args = parser.parse_args()
    infer(args)
