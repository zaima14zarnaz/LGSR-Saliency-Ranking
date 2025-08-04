import json
import re
import time
from llms.prompt_generator import PromptGenerator
from itertools import islice

class LabelInfuser:
    GEMINI_QUOTA = 500
    START_INDEX = 0
    def __init__(self, model, src_json, dest_json, images_dir, base_prompt_text, metadata_save_dir):
        self.model = model
        self.prompt_generator = PromptGenerator()
        self.prompt_generator.set_label_infuser_prompt_data(src_json=src_json, 
                                                            base_prompt_text=base_prompt_text, 
                                                            images_dir=images_dir,
                                                            metadata_save_dir=metadata_save_dir)
        self.dest_json = dest_json
        self.reformatted_desc = {}
    
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
    
    def infuse_labels(self):
        while True:
            text, image, index = self.prompt_generator.get_label_infuser_prompt()
            if text == None:
                self.save_new_desc()
                return
            print(text)
            response = self.model.get_response(text, image)
            if response is None:
                continue
            
            self.reformatted_desc[image] = response
            
            self.save_new_desc()

            print(f'Description generated for image no {index}: {self.reformatted_desc[image]}', flush=True)
            # if index % 15 == 0 and index != 0:
            #     time.sleep(65)

        
    def save_new_desc(self):
        with open(self.dest_json, "r") as f:
            old_data = json.load(f)
        
        old_data.update(self.reformatted_desc)
        with open(self.dest_json, "w") as f:
            json.dump(old_data, f, indent=2)
                