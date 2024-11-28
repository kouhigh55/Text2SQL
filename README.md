# Text2SQL
Leading LLMs to Text2SQL with Prompt-Based Fine-Tuning

### QuickStart

This project requires unsloth running environment, origin unsloth repository: [Unsloth](https://github.com/unslothai/unsloth)

Or, start by creating a conda environment:

```commandline
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

### Fine-tuning

To run fine-tuning of the Text2SQL model, run `experiments/unsloth_llama.py`

For example:
```commandline
python ./experiments/unsloth_infer.py --schema ./home/data/schema.json 
```
The schema file can be found in huggingface: richardr1126/spider-schema, or just retrive schema from Spider database.

### Inference
To run fine-tuning of the Text2SQL model, run `experiments/unsloth_infer.py`

```commandline
python ./experiments/unsloth_infer.py --schema ./home/data/schema.json
```

Notice that spider db should be put in `home` dir, specified by `db_dir` argument.

Spider can be found at [github.com/taoyds/spider](https://github.com/taoyds/spider)

Spider database can be found at [Google Drive](https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view)

