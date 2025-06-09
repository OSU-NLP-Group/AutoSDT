import json

data = [json.loads(x) for x in open("final_combined_training_data_expanded.jsonl")]
held_out_urls = set(json.load(open("held_out_file_urls_updated.json")))

SYSTEM_PROMPT = """You are an expert Python programming assistant that helps scientist users to write high-quality code to solve their tasks.
Given a user request, you are expected to write a complete program that accomplishes the requested task and save any outputs in the correct format.
Please wrap your program in a code block that specifies the script type, python. For example:
```python
print("Hello World!")
```
"""

output_data = []

cnt = 0
for x in data:
    if "REGULAR_EXPRESSION_ERROR" in x["instruction"] or \
         "REGULAR_EXPRESSION_ERROR" in x["program_code"] or \
         "REGULAR_EXPRESSION_ERROR" in x["file_url"]:
          continue
    if x["file_url"] in held_out_urls:
        print(f"Skipping {x['file_url']} as it is in the held out set")
        cnt += 1
        print("Count of held out files skipped: ", cnt)
        continue
    output_data.append({
         "instruction": SYSTEM_PROMPT,
         "input": x["instruction"],
         "output": f"```python\n{x['program_code']}\n```",
         "discipline": x["discipline"],
         "file_url": x["file_url"],
         "output_fname": x["output_fname"],
    })
import random
random.shuffle(output_data)

with open("data/final_combined_training_data_expanded.json", "w") as f:
    json.dump(output_data, f, indent=4)