import csv
import json
import re

# Function to read the CSV file and return a dictionary of function names and embeddings
def read_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        func_embeddings = {}
        for row in csv_reader:
            function_name = row[0]
            embeddings = row[1:]
            func_embeddings[function_name] = embeddings
        # import pprint;pprint.pprint(func_embeddings)
        return func_embeddings

# Function to read the log file and return a dictionary of function names and tokens
def read_log(file_path):
    cleaned_log_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(r"'([^']+)': \[([^]]*)\]", line)
            if match:
                full_function_name = match.group(1)
                tokens = match.group(2).strip().split(', ')
                # Remove the numeric prefix and the following period
                function_name = full_function_name.split('.', 1)[-1]
                cleaned_log_dict[function_name] = [token.strip("'") for token in tokens]
    return cleaned_log_dict

# Function to merge the data from both files and save as JSON
def merge_and_save_as_json(csv_data, log_data, output_file):
    merged_data = []
    for func_name, embeddings in csv_data.items():
        if func_name in log_data:
            merged_data.append({
                "function": func_name,
                "embeddings": embeddings,
                "tokens": log_data[func_name]
            })

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(merged_data, file, indent=4)
    # import pprint; pprint.pprint(merged_data)
# File paths (update these paths with your actual file locations)
csv_file_path = 'metrics/func_embeds_epoch_20.csv'
log_file_path = 'metrics/test_actual_kw_0.log'

# Process the files
csv_data = read_csv(csv_file_path)
log_data = read_log(log_file_path)
merge_and_save_as_json(csv_data, log_data, 'data/output.json')

print("JSON file created successfully.")