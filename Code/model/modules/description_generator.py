import json
import time
import os
from llms.prompt_generator import PromptGenerator

class DescriptionGenerator:
    GEMINI_QUOTA = 500
    START_INDEX = 5849
    def __init__(self, model, base_prompt_text, img_data_dir, description_dir, images_dir, img_count, human_anns_dir):
        self.model = model
        self.prompt_generator = PromptGenerator()
        self.prompt_generator.set_desc_generator_data(images_dir=images_dir, 
                                                      img_data_dir=img_data_dir, 
                                                      description_data_dir=description_dir, 
                                                      base_prompt_text=base_prompt_text,
                                                      img_count=img_count,
                                                      human_anns_dir=human_anns_dir)
        self.img_data_dir = img_data_dir
        self.description_dir = description_dir
        self.description_data = {}
        self.img_count = img_count
    
    def generator_descriptions(self):
        
        while True:
            text, image, index = self.prompt_generator.get_description_prompt()
            if index is None:
                # print(f'Index is none for image {image}')
                continue
            if index >= self.img_count:
                break
            if text == None:
                # self.save_descriptions()
                return

            response = self.model.get_response(text, image)
            if response is not None: 
                self.description_data[image] = response

                print(f'Description generated for image no: {index} \n Description: {response}', flush=True)
                self.save_descriptions()
            else:
                print(f'image {image} file not found')
                break

            # if index % 20 == 0 and index != 0:
            #     time.sleep(65)
    
    def generate_null_desc(self):
        descriptions = self.read_json(self.description_dir)
        while True:
            text, image, index = self.prompt_generator.get_description_prompt()
            if descriptions[image] is not None:
                continue
            if index >= self.img_count:
                break
            if text == None:
                self.save_descriptions()
                return
        
            response = self.model.get_response(text, image)
            self.description_data[image] = response

            print(f'Description generated for image no: {index} \n Description: {response}', flush=True)
            self.save_descriptions()

            if index % 20 == 0 and index != 0:
                time.sleep(65)


    
    def save_descriptions(self):
        with open(self.description_dir, "r") as f:
            old_images = json.load(f)

        # Merge dict2 into dict1
        old_images.update(self.description_data)

        # Write the updated dictionary back to the same file
        with open(self.description_dir, "w") as f:
            json.dump(old_images, f, indent=2)

    def read_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data