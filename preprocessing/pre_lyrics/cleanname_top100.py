import pandas as pd
import re

# Function to remove parentheses and the text within them
def remove_parentheses_and_space(text):
    return re.sub(r"\s*\([^)]*\)", "", text)

# Function to process the file
def process_file(input_file_path, output_file_path):
    df = pd.read_csv(input_file_path)

    df['track_name'] = df['track_name'].apply(remove_parentheses_and_space)

    df.to_csv(output_file_path, index=False)

# 2013-2023 processing
def main():
    for year in range(2013, 2024): 
        input_file_path = f'data/data_top100/data_top100_raw/{year}.csv'
        output_file_path = f'data/data_top100/data_top100_raw/{year}_processed_clean.csv'

        process_file(input_file_path, output_file_path)
        print(f'Processed file for year {year}')

if __name__ == "__main__":
    main()
