import json
import random
import os

def sample_aligned_json_data():
    file_path_1 = 'exp/nonthink/generated_answer.json'
    file_path_2 = 'exp/think/generated_answer.json'
    
    output_path_1 = 'exp/nonthink/generated_answer_.json'
    output_path_2 = 'exp/think/generated_answer_.json'
    
    target_count = 500
    print("Loading files...")
    
    if not os.path.exists(file_path_1) or not os.path.exists(file_path_2):
        print("Error: One or both input files do not exist.")
        return

    try:
        with open(file_path_1, 'r', encoding='utf-8') as f1:
            data_1 = json.load(f1)
        with open(file_path_2, 'r', encoding='utf-8') as f2:
            data_2 = json.load(f2)
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON.")
        return

    len1 = len(data_1)
    len2 = len(data_2)
    
    print(f"File 1 length: {len1}")
    print(f"File 2 length: {len2}")

    if len1 != len2:
        print("Error: The files have different lengths! Cannot perform aligned sampling.")
        return

    if len1 < target_count:
        print(f"Error: Total items ({len1}) is less than target ({target_count}).")
        return

    print(f"Generating {target_count} random indices...")
    selected_indices = random.sample(range(len1), target_count)
    print("Extracting data based on identical indices...")
    sampled_data_1 = [data_1[i] for i in selected_indices]
    sampled_data_2 = [data_2[i] for i in selected_indices]
    assert len(sampled_data_1) == target_count
    assert len(sampled_data_2) == target_count
    print("Assertion passed: Both lists have exactly 1000 items.")
    print(f"Saving to {output_path_1}...")
    with open(output_path_1, 'w', encoding='utf-8') as f1:
        json.dump(sampled_data_1, f1, indent=4)

    print(f"Saving to {output_path_2}...")
    with open(output_path_2, 'w', encoding='utf-8') as f2:
        json.dump(sampled_data_2, f2, indent=4)

    print("Success! Synchronized sampling complete.")

if __name__ == "__main__":
    sample_aligned_json_data()
