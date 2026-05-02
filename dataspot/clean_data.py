import os
import re
from pathlib import Path
from typing import List

# Import the renamed cleaner module
def get_data_paths(base_dir: Path) -> List[Path]:
    """
    Collects paths to text files from 'cleaned_data' subdirectory and 'training_data.txt'.
    """
    data_paths = []

    # Add DATA SPOT/training_data.txt
    data_paths.append(base_dir / "data.txt")

    # Add all .txt files from DATA SPOT/cleaned_data/
    cleaned_data_dir = base_dir / "cleaned_data"
    if cleaned_data_dir.is_dir():
        for file_path in cleaned_data_dir.glob("*.txt"):
            data_paths.append(file_path)
    
    return [p for p in data_paths if p.is_file()]

def process_file_content(content: str, filename: str) -> str:
    """
    Applies appropriate cleaning based on filename or content type.
    """
    # Apply initial aggressive HTML/JS/CSS removal
    content = remove_html_tags(content)
    content = remove_js_css(content)
    
    if "book" in filename.lower() or "gutenberg" in filename.lower():
        cleaned = clean_gutenberg(content)
    elif "web" in filename.lower() or "html" in filename.lower():
        cleaned = clean_web_text(content)
    else:
        # For general text, apply a robust generic cleaner
        cleaned = generic_clean(content)
        
    return cleaned

def main():
    root_dir = Path(__file__).resolve().parents[2] # Adjust to get to the /home/fillipus/Downloads/ftkAI/
    data_spot_dir = Path(__file__).parent # This is DATA SPOT/
    
    # Target output file in the root directory
    output_training_data_path = root_dir / "data.txt"

    # Get all relevant input data paths
    input_files = get_data_paths(data_spot_dir)
    
    if not input_files:
        print("No input data files found to clean.")
        return

    all_cleaned_content = []

    print(f"Cleaning data from: {data_spot_dir}")
    for file_path in input_files:
        print(f"  Processing file: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            cleaned_content = process_file_content(content, file_path.name)
            if cleaned_content:
                all_cleaned_content.append(cleaned_content)
        except Exception as e:
            print(f"  Error cleaning {file_path.name}: {e}")

    # Write all cleaned content to the root training_data.txt
    # We will overwrite the file to ensure it's a fresh combination
    if all_cleaned_content:
        combined_text = "\n".join(all_cleaned_content)
        try:
            output_training_data_path.write_text(combined_text, encoding="utf-8")
            print(f"Successfully cleaned and combined data into: {output_training_data_path}")
        except Exception as e:
            print(f"Error writing to {output_training_data_path}: {e}")
    else:
        print(f"No cleaned content to write to {output_training_data_path}.")

if __name__ == "__main__":
    main()
