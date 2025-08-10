cd ../../

# uncomment the following line to clone the environments if this is the first time running the script
# bash autosdt/scripts/clone_envs.sh

python autosdt/src/autosdt_adapt_modify_code.py \
--llm_engine_name us.anthropic.claude-3-7-sonnet-20250219-v1:0 \
--input_jsonl autosdt/result/ready_python_files.jsonl \
--output_jsonl autosdt/result/coding_agent_result.jsonl \
--program_output_dir generated_gold_programs \
--benchmark_dir benchmark/datasets \
--api_version 2024-12-01-preview \
--num_threads 8 \
--use_self_debug
