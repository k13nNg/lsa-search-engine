import json
from tqdm import tqdm

INPUT_PATH="./data/arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = "./data/arxiv-metadata-filtered.json"


# filter only papers that relate to Numerical Analysis, Cryptography, AI, ML, Combinatorics, General Mathematics, Logic, and Optimization and Control for ease of storage
FILTER = {"cs.AI", "cs.CR", "cs.LG", "cs.NA", "math.CO", "math.LO", "math.OC"}
def generate_math_cs_json():
    print("Generation process started")
    total_count = 0
    kept_count = 0

    with open(INPUT_PATH, 'r') as f_in, open (OUTPUT_PATH, 'w') as f_out:
        for line in tqdm(f_in, total=2890332, unit="docs"):
            try:
                doc = json.loads(line)
                categories = set(doc['categories'].split())

                # only record papers that match the pre-defined categories
                if not(categories.isdisjoint(FILTER)):
                    mini_doc = {
                        'id': doc['id'],
                        'title': doc['title'].replace('\n', ' ').strip(),
                        'abstract': doc['abstract'].replace('\n', ' ').strip(),
                        'categories': doc['categories']
                    }
                    kept_count += 1
                    f_out.write(json.dumps(mini_doc) + '\n')
                total_count += 1
            except Exception as e:
                print(f"Exception: {e}")
                continue

    print(f"Done! Filtered from {total_count} papers down to {kept_count} papers")
    print(f"Saved output to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_math_cs_json()