import pandas as pd
import json
from pathlib import Path
import argparse

def load_kaggle_data(file_path):
    """Load the Kaggle dataset from CSV."""
    return pd.read_csv(file_path)

def format_for_tinyllama(df):
    """Format the data for TinyLlama fine-tuning."""
    formatted_data = []
    
    for _, row in df.iterrows():
        # Format the prompt with the type and question
        prompt = f"### User:\nType: {row['type']}\nQuestion: {row['question']}\n\n### Assistant:"
        
        formatted_data.append({
            "prompt": prompt,
            "response": row['response']
        })
    
    return formatted_data

def save_to_jsonl(data, output_path):
    """Save the formatted data to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Prepare data for TinyLlama fine-tuning')
    parser.add_argument('--input', required=True, help='Path to the input CSV file')
    parser.add_argument('--output_dir', default='processed_data', help='Directory to save processed data')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load and process the data
    print("Loading data...")
    df = load_kaggle_data(args.input)
    
    print("Formatting data for TinyLlama...")
    formatted_data = format_for_tinyllama(df)
    
    # Save as JSONL
    jsonl_path = output_dir / 'tinyllama_training_data.jsonl'
    print(f"Saving to {jsonl_path}...")
    save_to_jsonl(formatted_data, jsonl_path)
    
    print("Data preparation complete!")
    print(f"Total examples: {len(formatted_data)}")
    print(f"Data types distribution: {df['type'].value_counts().to_dict()}")
    
    # Print a sample of the formatted data
    print("\nSample of formatted data:")
    print(json.dumps(formatted_data[0], indent=2))

if __name__ == "__main__":
    main() 