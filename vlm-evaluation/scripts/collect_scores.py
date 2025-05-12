import os
import json
import argparse
import csv
from glob import glob

# mapping from eval identifier (lowercase substring) to the exact summary key
SPECIAL_KEYS = {
    "ai2d": "accuracy__AI2D-test-Accuracy",
    "pope": "accuracy__POPE-final-Accuracy",
    "text-vqa": "accuracy__TextVQA-OCR",
}

def extract_accuracy_metrics(root_dir, model_id):
    # Find all metrics.json files in subdirectories
    metrics_files = glob(os.path.join(root_dir, "**", "metrics.json"), recursive=True)
    
    results = {}

    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            
            if "summary" not in data:
                continue
            
            summary = data["summary"]
            # derive eval name from its directory
            eval_subdir = os.path.relpath(os.path.dirname(metrics_file), root_dir)
            eval_key = eval_subdir.lower()
            
            # try special keys first
            accuracy_value = None
            for tag, special_key in SPECIAL_KEYS.items():
                if tag in eval_key and special_key in summary:
                    accuracy_value = summary[special_key]
                    break
            
            # fallback to generic "accuracy"
            if accuracy_value is None and "accuracy" in summary:
                accuracy_value = summary["accuracy"]
            
            if accuracy_value is not None:
                results[eval_subdir] = accuracy_value

        except Exception as e:
            print(f"Error processing {metrics_file}: {e}")
    
    return results

def save_to_csv(results, model_id, output_file=None):
    if output_file is None:
        output_file = f"{model_id}_accuracy_results.csv"
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['eval_name', 'accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for eval_name, accuracy in results.items():
            writer.writerow({'eval_name': eval_name, 'accuracy': accuracy})
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Extract accuracy metrics from evaluation results')
    parser.add_argument('--root_dir', help='Root directory containing evaluation results', default="/fsx/users/haoli/datopenqwen/results")
    parser.add_argument('--model_id', help='Model ID to extract metrics for', default="qwen_vlm_dp")
    parser.add_argument('--output_path', help='Output CSV directory path', default="/fsx/users/haoli/datopenqwen/results")
    args = parser.parse_args()
    
    results = extract_accuracy_metrics(args.root_dir, args.model_id)
    
    output_file = save_to_csv(
        results,
        args.model_id,
        os.path.join(args.output_path, f"{args.model_id}_accuracy_results.csv")
    )
    
    print(f"Accuracy results for model {args.model_id} saved to {output_file}")
    print(f"Found {len(results)} evaluation results")

if __name__ == "__main__":
    main()