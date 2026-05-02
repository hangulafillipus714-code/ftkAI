# DATA SPOT

This directory contains data sources and scripts for cleaning and preparing data for model training.

## Data Cleaning Utility

The `clean_data.py` script in this directory is a powerful utility designed to process raw text data, remove noise, and prepare it for training the language model. It leverages `cleaner.py` for comprehensive cleaning operations.

### How to Use `clean_data.py`

This script will read text files from `DATA SPOT/cleaned_data/` and `DATA SPOT/training_data.txt`, apply cleaning rules, and then combine all the cleaned content into the main `training_data.txt` file in the project's root directory (`/home/fillipus/Downloads/ftkAI/training_data.txt`).

**To clean your data and update the main `training_data.txt`:**

1.  **Place your raw text files:**
    *   For new data you want to clean, place your `.txt` files into the `DATA SPOT/cleaned_data/` directory.
    *   The `DATA SPOT/training_data.txt` file itself will also be cleaned and included.
2.  **Run the cleaning script:**
    *   Open your terminal in the project root directory (`/home/fillipus/Downloads/ftkAI/`).
    *   Execute the script using the following command:
        ```bash
        python "DATA SPOT/clean_data.py"
        ```
    *   The script will print its progress and confirm when the `training_data.txt` in the root directory has been updated.

**Important Notes:**

*   **Overwriting `training_data.txt`**: The script will **overwrite** the `training_data.txt` file in the project's root directory with the newly cleaned and combined content. Ensure you have backups if you wish to preserve previous versions.
*   **Filename-based Cleaning**: The script attempts to apply specific cleaning rules based on the filename:
    *   Files containing "book" or "gutenberg" in their name will get Project Gutenberg-specific cleaning.
    *   Files containing "web" or "html" will get web-text specific cleaning.
    *   All other files receive general-purpose text cleaning.
*   **Growing your dataset**: To continuously add new, clean data to your model's training set, simply place new raw `.txt` files in `DATA SPOT/cleaned_data/` and re-run the `clean_data.py` script.

Training loss ↓
Validation loss ↑

→ overfitting started → STOP.