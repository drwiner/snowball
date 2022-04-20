
import json
class JsonHelper:

    @staticmethod
    def parse_json(path_with_json):
        with open(path_with_json) as filename:
            try:
                data = json.load(filename)
            except json.decoder.JSONDecodeError:
                print("ERROR: failed to parse " + path_with_json)
                return None
        return data

    @staticmethod
    def write_json(dict_for_json, file_name):
        with open(file_name, 'w') as filename:
            json.dump(dict_for_json, filename, ensure_ascii=False, indent=4)
