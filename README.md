# AutoSDT
This is the official codebase of AutoSDT:
Scaling Data-Driven Discovery Tasks Toward Open Co-Scientists.

## AutoSDT-Pipeline
### Configure Azure endpoint and API key
```python
vim ~/.bashrc
export AZURE_OPENAI_KEY=YOUR_AZURE_API_KEY
export AZURE_ENDPOINT=YOUR_AZURE_ENDPOINT
export AZURE_API_VERSION=YOUR_AZURE_API_VERSION
source ~/.bashrc
```

### AutoSDT-Crawl: Download repos to local dir and extract py files
```python
cd CodingAgent/scripts
bash run_crawl_scientific_py_files.sh
```

### AutoSDT-Select-Step1: Model-based scientific task verification
```python
bash run_scientific_task_verify.sh
```

### AutoSDT-Select-Step2: Model-based dataset locating
```python
bash run_locate_dataset.sh
```

### AutoSDT-Select-Step3: Move the datasets to the right place
```python
bash run_prepare_dataset_env.sh
```

### AutoSDT-Adapt
```python
cd ../..
bash CodingAgent/scripts/clone_envs.sh
bash run_coding_agent.sh
cd CodingAgent/scripts
bash run_gen_inst.sh
```

After the above steps, you should obtain a `final_combined_training_data.jsonl` containing the generated instructions and code. After that, run
```python
python convert_data_to_alpaca_format.py
```
to convert the data format into alpaca training format.

## Supervised Fine-tuning
We use the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) library to conduct SFT experiments. We provide the `.yaml` files within this repo:

```python
-- qwen2.5-coder-7b-instruct_full_sft.yaml
-- qwen2.5-coder-7b-instruct_full_sft.yaml
-- qwen2.5-coder-7b-instruct_full_sft.yaml
```
Please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for more details.

## Inference and Evaluation
For ScienceAgentBench, we directly follow the original repo for running inference and evaluation. Please refer to `ScienceAgentBench/README.md` for more information.

For DiscoveryBench, first start an LLM engine at localhost using [vllm](https://docs.vllm.ai/en/latest/), then run
```python
python evaluate_with_llm_engine.py
```
to generate all the evaluation results, and run
```python
python cal_eval_avg.py
```
to compute the final results.
