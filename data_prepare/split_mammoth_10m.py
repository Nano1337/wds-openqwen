import json
import os
from tqdm import tqdm
import jsonlines
import multiprocessing as mp

n_patches = 729

INPUT_JSON = "/fsx/data/common/MAmmoTH-VL-Instruct-12M/mammoth_si_10M.json"
SIMPLE_JSONL = "/fsx/data/common/MAmmoTH-VL-Instruct-12M/mammoth_si_10M_simple.jsonl"
OUTPUT_DIR = "/fsx/data/common/MAmmoTH-VL-Instruct-12M/instruction_si_split"

def process_sample(sample):
    if "image" in sample:
        path = sample["image"].replace("/", "_").replace(".", "_")[:200] + ".json"
    else:
        path = (str(sample["source"]) + str(sample["id"])).replace(".", "_") + ".json"
        if os.path.exists(f"{OUTPUT_DIR}/{path}"):
            return None
    if not path.endswith(".json"):
        print(path)
        return None
    n_words = n_patches + sum(len(turn["value"].replace("<image>", "").split()) for turn in sample["conversations"])
    simple_sample = {"path": path, "length": n_words}
    with open(f"{OUTPUT_DIR}/{path}", "w") as f:
        f.write(json.dumps(sample))
    return simple_sample

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading {INPUT_JSON}")
    data = json.load(open(INPUT_JSON))
    with jsonlines.open(SIMPLE_JSONL, 'w') as writer:
        num_workers = mp.cpu_count()
        print(f"Using {num_workers} workers")
        with mp.Pool(processes=num_workers) as pool:
            for simple_sample in tqdm(pool.imap(process_sample, data), total=len(data), desc="Processing samples"):
                if simple_sample is not None:
                    writer.write(simple_sample)

if __name__ == "__main__":
    main()