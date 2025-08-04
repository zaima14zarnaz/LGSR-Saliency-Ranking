import json
import re
import time
from llms.prompt_generator import PromptGenerator
from itertools import islice

class RankGenerator:
    GEMINI_QUOTA = 500
    START_INDEX = 0
    def __init__(self, model, src_json, dest_json, base_prompt_text):
        self.model = model
        self.prompt_generator = PromptGenerator()
        self.prompt_generator.set_rank_prompt_data(src_json=src_json, rank_data_dir=dest_json, base_prompt_text=base_prompt_text)
        self.dest_json = dest_json
        self.rank_data = {}
    
    def transform_to_rank_data(self, response):
        result = {}

        # Matches tuples like: (person_3, 7.6)
        # Label can contain spaces and underscores, score must be numeric
        pattern = re.compile(r"\(\s*(.+?)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\s*\)")

        matches = pattern.findall(response)

        # Count number of '(' to ensure all tuples matched
        expected_count = response.count('(')
        if len(matches) != expected_count:
            return {}

        # Build dictionary
        for label, score in matches:
            result[label.strip()] = float(score)

        return result
    
    def find_ranks(self):
        while True:
            text, image, index = self.prompt_generator.get_rank_prompt()

            if text == None:
                self.save_ranks()
                return

            response = self.model.get_response(text)
            if response is None:
                continue
            
            rank_data = self.transform_to_rank_data(response=response)
            self.rank_data[image] = rank_data
            
            self.save_ranks()

            print(f'Description generated for image no {index}: {self.rank_data[image]}', flush=True)
            # if index % 15 == 0 and index != 0:
            #     time.sleep(65)

        
    def save_ranks(self):
        with open(self.dest_json, "r") as f:
            old_data = json.load(f)
        
        old_data.update(self.rank_data)
        with open(self.dest_json, "w") as f:
            json.dump(old_data, f, indent=2)
                