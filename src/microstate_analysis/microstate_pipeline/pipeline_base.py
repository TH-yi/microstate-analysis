from microstate_analysis.logger.dualhandler import DualHandler
import os
import json


class PipelineBase():
    def ensure_dir(self, dir):
        directory = os.path.dirname(dir)
        if not os.path.exists(directory):
            os.makedirs(directory)
            self.logger.log_warning("Dir not exist, created:", str(directory))

    def dump_to_json(self, json_data, output_dir, file_name):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                self.logger.log_info(f"Output directory {output_dir} created.")

            if not file_name.lower().endswith(".json"):
                file_name = f"{file_name}.json"

            json_file_path = os.path.join(output_dir, file_name)

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

            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)

            self.logger.log_info(f"Successfully dumped JSON data to {json_file_path}")

        except Exception as e:
            self.logger.log_error(f"Failed to dump JSON data to {json_file_path}: {str(e)}")

