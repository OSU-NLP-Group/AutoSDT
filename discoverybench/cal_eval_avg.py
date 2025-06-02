import os
import json

MODEL_NAME = "qwen32b_0shot_0516_run4"
EVAL_LOGS_DIR = "eval_results_{}".format(MODEL_NAME)

def extract_last_json(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    for i in range(len(lines)-1, -1, -1):
        if lines[i].strip().startswith("{"):
            json_str = "".join(lines[i:]).strip()
            try:
                return json.loads(json_str)
            except Exception:
                continue
    return None

all_scores = []
zero_scores = 0

for fname in os.listdir(EVAL_LOGS_DIR):
    if not fname.endswith(".json"):
        continue
    fpath = os.path.join(EVAL_LOGS_DIR, fname)
    result = extract_last_json(fpath)
    if result is not None and "final_score" in result:
        score = result["final_score"]
        all_scores.append(score)
        if score == 0.0:
            zero_scores += 1
    else:
        print(f"{fname} not found final_score")

if all_scores:
    print(f"Total files: {len(all_scores)}")
    print(f"Total score: {sum(all_scores)}")
    print(f"Avg score: {sum(all_scores)/len(all_scores):.4f}")
    print(f"final_score == 0.0 count: {zero_scores}")  # 新增
    # print(sorted(all_scores))
else:
    print("No scores found.")