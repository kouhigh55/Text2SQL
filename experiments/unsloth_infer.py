# -*- coding:utf-8 -*-　
# Last modify: Liu Wentao
# Description: Infer unsloth model
# Note:

import argparse
import os
import re
import sys
import json
import torch

import pandas as pd

sys.path.append(os.path.split(sys.path[0])[0])

from unsloth import FastLanguageModel, get_chat_template
from tqdm import tqdm
from utils import get_sqls, process_duplication

prompt_base = """
Turn this natural language query into standard SQL language.
query: {ques},
table_schema: {sche},
SQL result:
"""


def construct_prompt(question, schema):
    return prompt_base.format(ques=question, sche=schema)


def infer(args):

    torch.manual_seed(args.seed)

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
            num_return_sequences=args.self_consistent_n
        )
        if args.self_consistent_n == 1:
            inference_output = tokenizer.batch_decode(outputs)
            real_output = extract_from_template(inference_output[0])

            predicted_ls.append(real_output)

            if args.show_infer:
                print(f"Input: {messages}\n"
                      f"Generated: {real_output}\n")
        else:
            inference_outputs = tokenizer.batch_decode(outputs)
            list_output_text = [extract_from_template(output) for output in inference_outputs]
            processed_sqls = []
            for sql in list_output_text:
                sql = " ".join(sql.replace("\n", " ").split())
                sql = process_duplication(sql)
                if sql.startswith("SELECT"):
                    pass
                elif sql.startswith(" "):
                    sql = "SELECT" + sql
                else:
                    sql = "SELECT " + sql
                processed_sqls.append(sql)
            result = {
                'db_id': row['db_id'],
                'p_sqls': processed_sqls
            }
            final_sqls = get_sqls(
                results=[result],
                select_number=args.self_consistent_n,
                db_dir=args.db_dir
            )
            predicted_ls.append(final_sqls[0])
            if args.show_infer:
                processed_sqls_str = "\n\t".join(processed_sqls)
                print(f"Input: {messages}\n"
                      f"select from:\n\t{processed_sqls_str}\n"
                      f"Generated:\n\t{final_sqls[0]}\n")

    with open(os.path.join(args.save_dir), "w") as file:
        for line in predicted_ls:
            file.write(line + "\n")


def extract_from_template(inference_output):
    output_text = re.search(
        r'<\|start_header_id\|>assistant<\|end_header_id\|>\n\n(.*?)<\|eot_id\|>',
        inference_output,
        re.DOTALL,
    )
    if output_text:
        real_output = output_text.group(1).strip()
    else:
        real_output = ""
        print(f"Error when parsing real output from {inference_output[0]}")
    return real_output


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

    # Torch.manual_seed(3407) is all you need
    parser.add_argument(
        '--seed',
        '-s',
        type=int,
        required=False,
        default=3407,
        help='Random seed for reproducibility',
    )

    parser.add_argument('--max_seq_length', type=int, required=False, default=512)
    parser.add_argument('--load_in_4bit', type=bool, required=False, default=True,
                        help='Use 4bit quantization to reduce memory usage. Can be False')
    parser.add_argument('--show_infer', type=bool, required=False, default=False)
    parser.add_argument('--batch_size', type=int, required=False, default=1)
    parser.add_argument('--self_consistent_n', type=int, required=False, default=1)
    parser.add_argument('--db_dir', type=str, required=False, default='./home/data/spider_data/test_database')

    args = parser.parse_args()
    infer(args)
