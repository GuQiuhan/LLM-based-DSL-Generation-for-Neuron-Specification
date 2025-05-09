import json
from datetime import datetime

from request import TGIClient

# @qiuhan: update the IP
MODEL_ENDPOINTS = {
    "gemma-7b": "http://10.192.122.120:8082",
    "deepseek-6.7b": "http://10.192.122.120:8083",
    "llama4-17B": "http://10.192.122.120:8084",
    "llama3-1B": "http://10.192.122.120:8085",
    "llama3-70B": "http://10.192.122.120:8086",
}


def run_multi_model_comparison(prompt: str, save_path: str = None):
    print(f"\n**********\n📌 Prompt:\n{prompt}\n**********\n")
    results = {}

    for name, url in MODEL_ENDPOINTS.items():
        print(f"🚀 Querying {name} at {url} ...")
        client = TGIClient(model=url, max_new_tokens=512, temperature=0.7)
        output = client.textgen(prompt)
        print(f"✅ {name} Output:\n{output}\n{'-'*60}")
        results[name] = output

    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"comparison_results_{timestamp}.json"

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {"prompt": prompt, "results": results}, f, indent=2, ensure_ascii=False
        )

    print(f"\n**********Results saved to {save_path}**********\n")


if __name__ == "__main__":
    prompt = "Write a DSL abstract transformer for ReLU(x) over [l, u]. Ensure it is sound and tight."
    run_multi_model_comparison(prompt)
