import os
import json
from PIL import Image, ImageDraw
import scipy.io
import sys
import re
from Code.model.module_helpers.image_segmentation import ImageSegmentation
from Code.model.module_helpers.eyegaze import EyeGazeAnalyzer




class Pipeline:
    class InstanceInfo:
        def __init__(self, label, bbox, score):
            self.label = label
            self.bbox = bbox
            self.score = score

        def to_dict(self):
            return {
                "label": self.label,
                "score": self.score,
                "bbox": self.bbox
            }

    def __init__(self, coco_annotations, object_extractor):
        self.image_to_mat_files = {}
        self.json_list = {}
        self.object_extractor = object_extractor
        self.eyegazeAnalyzer = EyeGazeAnalyzer(segmentor=self.object_extractor)

    def read_directory(self, dir_path):
        file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        return file_list
    
    def match_images_to_mat_files(self, img_count, image_files, fixation_files, gaze=False):
        if not gaze:
            self.image_to_mat_files = {fname:fname for fname in human_anns.keys()}
            return
        fixation_fnames = [os.path.splitext(f)[0] for f in fixation_files]
        base_filenames = []
        for f_image in image_files:
            image_fname = os.path.splitext(f_image)[0] 
            index_fixation_list = fixation_fnames.index(image_fname)
            self.image_to_mat_files[f_image] = fixation_files[index_fixation_list]
            base_filenames.append(image_fname)
        # self.object_extractor.load_annotations(image_fname)

    def get_annotations(self, image_fname, image_addr):
        instances, obj_list, annotations = self.object_extractor.label_instances(image_fname)
        return instances, annotations, obj_list


    def draw_bbox(self, draw, instance):
        bbox = [round(instance.bbox[0],2), 
                        round(instance.bbox[1],2), 
                        round(instance.bbox[0]+instance.bbox[2],2),
                        round(instance.bbox[1]+instance.bbox[3],2)]
        draw.rectangle(bbox, outline="red", width=1)

        # text_position = (bbox[0], bbox[1] - 10)
        # draw.text(text_position, instance.label, fill="red")

    def rank_objects_by_eyegaze(self, max_items, image_fname, img_addr, fixation_addr, gaze=False):
        instances, annotations, obj_list = self.get_annotations(image_fname=image_fname, image_addr=img_addr)
        if instances == None:
            return None
            
        if 'test' not in image_fname and gaze:
            self.eyegazeAnalyzer.set_data(annotations=annotations, img_fname=image_fname, img_addr=img_addr, fixation_addr=fixation_addr)
            sorted_objects, object_scores = self.eyegazeAnalyzer.rank_objects(instances)
            object_scores = dict(sorted(object_scores.items(), key=lambda item: -item[1]))
        else:
            object_scores = {object:-1 for label,object in obj_list.items()}
            
        boxed_image = Image.open(img_addr).convert("RGB")
        draw = ImageDraw.Draw(boxed_image)

        items = 0
        gaze_sal_scores = {}

        for instance, score in object_scores.items():
            if instance.label == 'bg':
                continue
            gaze_sal_scores[instance.label] = {"object":instance, "score":score}
            self.draw_bbox(draw, instance)

            items += 1
            if max_items is not None and items >= max_items:
                break
        
        return boxed_image, gaze_sal_scores 
    
    def transform_to_rank_data(self, response):
        result = {}
        pattern = re.compile(r"\(\s*(.+?)\s*,\s*([+-]?[0-9]*\.?[0-9]+)\s*\)")
        matches = pattern.findall(response)
        expected_count = response.count('(')
        if len(matches) != expected_count:
            return {}
        for label, score in matches:
            if '_' not in label:
                continue
            result[label.strip()] = float(score)
        return result
    
    def generate_sal_map(self, image_path, object_anns, saliency_scores):
        image = Image.open(image_path)
        width, height = image.size

        sal_map = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(sal_map)
        sorted_labels = [label for label, _ in saliency_scores.items()]

        for i, label in enumerate(sorted_labels):
            if label not in object_anns:
                continue
            gray_value = int((len(sorted_labels) - i) / len(sorted_labels) * 255)
            segmentations = object_anns[label]['object'].segmentations
            for seg in segmentations:
                polygon = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
                draw.polygon(polygon, fill=gray_value)

        return sal_map
        
    
    def pipeline(self, 
                img_fnames, 
                img_dir, 
                fixation_dir,
                saliency_map_dir, 
                desc_save_dir,
                llm_model, 
                desc_prompt_generator, 
                rank_prompt_generator,
                max_items=8):
        for img_fname in img_fnames:
            img_path = os.path.join(img_dir, img_fname)
            fixation_map_path = os.path.join(fixation_dir, os.path.splitext(img_fname)[0])
            # Module 1: Gaze-Based Saliency Extractor
            boxed_image, gaze_sal_scores = self.rank_objects_by_eyegaze(max_items=max_items, 
                                                            image_fname=img_fname, 
                                                            img_addr=img_path, 
                                                            fixation_addr=fixation_map_path,
                                                            gaze=True)

            # Module 2: Description Generator
            desc_prompt = desc_prompt_generator.get_description_prompt(gaze_sal_scores)
            img_desc = llm_model.get_response(desc_prompt, boxed_image)
            if img_desc is None:
                print(f'Unable to generate image description for image {img_fname}')
            self.save_json(desc_save_dir, {img_fname:img_desc})

            # Module 3: Rank Generator
            rank_prompt = rank_prompt_generator.get_rank_prompt(img_desc)
            saliency_scores_text_format = llm_model.get_response(rank_prompt)
            if saliency_scores_text_format is None:
                print(f'Unable to generate saliency scores for objects in image {img_fname}')
            saliency_scores = self.transform_to_rank_data(saliency_scores_text_format)
            
            # Generate saliency maps
            sal_map = self.generate_sal_map(img_path, gaze_sal_scores, saliency_scores)
            sal_map.save(os.path.join(saliency_map_dir, f'{os.path.splitext(img_fname)[0]}.png'))

            print(f'Generated saliency maps for image: {img_fname}')
            



    def store_instance(self, img_fname, gaze_sal_scores):
        self.json_list[img_fname] = gaze_sal_scores
    

    def read_json(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data
        
    def save_json(self, json_file, json_list, overwrite=False):
        if overwrite is True:
            with open(json_file, "w") as f:
                json.dump(json_list, f, indent=2)
        else:
            with open(json_file, "r") as f:
                old_data = json.load(f)

            # Merge dict2 into dict1
            old_data.update(json_list)

            # Write the updated dictionary back to the same file
            with open(json_file, "w") as f:
                json.dump(old_data, f, indent=2)
    





