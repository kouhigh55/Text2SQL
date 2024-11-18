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


def infer(args):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    # Load Spider schema
    with open(args.schema, "r", encoding="utf-8") as file:
        json_data = json.load(file)

    schema_dict = {entry["db_id"]: entry["Schema (values (type))"] for entry in json_data}

    # Load Spider dataset
    splits = {'train': 'spider/train-00000-of-00001.parquet', 'validation': 'spider/validation-00000-of-00001.parquet'}
    df = pd.read_parquet("hf://datasets/xlangai/spider/" + splits["validation"])

    predicted_ls = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing"):
        # Prepare messages for input
        messages = [
            {"role": "user", "content": construct_prompt(row['question'], schema_dict[row['db_id']]),},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,  # Must add for generation
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=args.max_seq_length,
            use_cache=True,
            temperature=1.5,
            min_p=0.1,
        )

        inference_output = tokenizer.batch_decode(outputs)
        output_text = re.search(
            r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
            inference_output[0],
            re.DOTALL,
        )

        if output_text:
            real_output = output_text.group(1).strip()
        else:
            real_output = ""
            print(f"Error when parsing real output from {inference_output[0]}")

        predicted_ls.append(real_output)

        if args.show_infer:
            print(f"Input: {messages}\n"
                  f"Generated: {real_output}\n")

    with open(os.path.join(args.save_dir), "w") as file:
        for line in predicted_ls:
            file.write(line + "\n")

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
        required=False,
        default='./home/output/results.txt',
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
