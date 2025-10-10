import numpy as np
import pandas as pd
import re
import json
import os
import csv

class QualityBase():
    def __init__(self):
        pass

    def get_first_json_path_in_directory(self, dir_path):
        try:
            files = os.listdir(dir_path)
            json_files = [f for f in files if f.endswith('.json')]

            if not json_files:
                raise FileNotFoundError

            first_json_file_path = os.path.join(dir_path, json_files[0])

            return first_json_file_path

        except Exception as e:
            raise e

    def read_first_json_in_directory(self, dir_path):
        try:
            files = os.listdir(dir_path)
            json_files = [f for f in files if f.endswith('.json')]

            if not json_files:
                raise FileNotFoundError

            first_json_file_path = os.path.join(dir_path, json_files[0])

            with open(first_json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)

            return data

        except Exception as e:
            raise e

    @staticmethod
    def calculate_list_std(list):
        return np.std(list)

    @staticmethod
    def contains_letter(s):
        if not isinstance(s, str):
            return False
        return bool(re.search(r'[a-df-zA-Z]', s))

    def is_index(self, lst):
        # Check if all elements are integers
        if all(isinstance(x, int) for x in lst):
            return True

        for x in lst:
            if self.contains_letter(x):
                return True

        return False

    def load_csv_with_index_check(self, csv_file):
        """
        Load a CSV file and handle it appropriately depending on whether
        it contains an index row and/or index column.
        Prints detailed information including the file name for traceability.
        """
        # Read the first few lines to detect potential index rows/columns
        sample_data = pd.read_csv(csv_file, nrows=6)

        # Check if the first column is an index column
        first_column = sample_data.iloc[1:5, 0].tolist()
        has_index_column = self.is_index(first_column)

        # Check if the first row is an index row
        first_row = sample_data.columns.tolist()[1:]
        has_index_row = self.is_index(first_row)

        if has_index_row and has_index_column:
            self.logger.log_info(f"File {csv_file} contains both index row and index column. Reading with both.")
            return pd.read_csv(csv_file, index_col=0, header=0)

        # If only an index column is detected
        if has_index_column:
            self.logger.log_info(f"File {csv_file} contains an index column. Reading with index column.")
            return pd.read_csv(csv_file, index_col=0)

        # If only an index row is detected
        elif has_index_row:
            self.logger.log_info(f"File {csv_file} contains an index row. Reading with index row.")
            return pd.read_csv(csv_file, header=0)

        else:
            # No index row/column detected – read in plain mode
            self.logger.log_info(f"File {csv_file} has no index row or column. Reading without index.")
            return pd.read_csv(csv_file, header=None)

    def ensure_dir(self, dir):
        directory = os.path.dirname(dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.logger.log_warning("Dir not exist, created:", str(directory))

    def _convert_keys_to_str(self, data):
        if isinstance(data, dict):
            return {str(k): self._convert_keys_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_keys_to_str(item) for item in data]
        else:
            return data

    def dump_to_json(self, json_data, output_dir, file_name):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.log_info(f"Output directory {output_dir} created.")

            json_file_path = os.path.join(output_dir, f'{file_name}.json')

            json_data = self._convert_keys_to_str(json_data)

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, separators=(',', ':'), sort_keys=True, ensure_ascii=False, indent=4)

            self.logger.log_info(f"Successfully dumped JSON data to {json_file_path}")

        except Exception as e:
            self.logger.log_error(f"Failed to dump JSON data: {str(e)}")

    def dump_to_json_path(self, json_data, json_file_path):
        try:
            output_dir = os.path.dirname(json_file_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.log_info(f"Output directory {output_dir} created.")

            json_data = self._convert_keys_to_str(json_data)

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)

            self.logger.log_info(f"Successfully dumped JSON data to {json_file_path}")

        except Exception as e:
            self.logger.log_error(f"Failed to dump JSON data to {json_file_path}: {str(e)}")


    @staticmethod
    def find_condition_name(task_type, condition_dict):
        for condition, tasks in condition_dict.items():
            if task_type in tasks:
                return condition
        return None

    def read_json(self, file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            self.logger.log_info(f"Successfully loaded JSON file: {file_path}")
            return data
        except Exception as e:
            self.logger.log_error(f"Error loading JSON file {file_path}: {str(e)}")
            return None

    def read_csv(self, file_path):
        try:
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                data = np.array(list(reader)).astype(float)
            self.logger.log_info(f"Successfully loaded CSV file: {file_path}")
            return data
        except Exception as e:
            self.logger.log_error(f"Error loading CSV file {file_path}: {str(e)}")
            return None