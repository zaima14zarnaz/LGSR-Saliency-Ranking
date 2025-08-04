import json
from itertools import islice
import os
import random
from scipy.stats import spearmanr
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import numpy as np
import cv2

class PromptGenerator:
    class Prompt:
        def __init__(self, text, image):
            self.text = text
            self.image = image
            
    def __init__(self, base_prompt_text):
        self.index = 0
        self.base_prompt_text = base_prompt_text
        self.images = []
        self.bbox_list = []
        self.desc = []
        self.total_img_count = 0
        self.COCO_INSTANCE_CATEGORY_NAMES = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "trafficlight", "firehydrant", "streetsign", "stopsign", "parkingmeter", "bench",
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
            "hat", "backpack", "umbrella", "shoe", "eyeglasses", "handbag", "tie", "suitcase",
            "frisbee", "skis", "snowboard", "sportsball", "kite", "baseballbat", "baseballglove",
            "skateboard", "surfboard", "tennisracket", "bottle", "plate", "wineglass", "cup", "fork",
            "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hotdog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "mirror",
            "diningtable", "window", "desk", "toilet", "door", "tvmonitor", "laptop", "mouse",
            "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddybear",
            "hairdrier", "toothbrush", "hairbrush"
        ]


    
    def set_desc_generator_data(self, 
                                images_dir, 
                                img_data_dir, 
                                description_data_dir, 
                                base_prompt_text,
                                img_count,
                                human_anns_dir):
        self.base_prompt_text = base_prompt_text
        self.index = 0
        self.img_count = img_count
        if img_data_dir is not None:
            with open(img_data_dir, "r") as f:
                img_data_dict = json.load(f)
            with open(human_anns_dir, "r") as f:
                human_anns_dir = json.load(f)
            with open(description_data_dir, "r") as f:
                existing_desc = json.load(f)
            
            # img_data_dict = self.scramble_dict(img_data_dict)
            files = os.listdir(images_dir)
                         

            filtered_bbox_data = {}
            print(f'Img dict len: {len(img_data_dict.keys())}')
            print(f'existing files: {len(existing_desc)}')
            for img, img_data in img_data_dict.items():
                if len(filtered_bbox_data) >= self.img_count:
                    break
                if existing_desc.get(img) is None and human_anns_dir.get(img) is not None and img in files:
                    filtered_bbox_data[img] = img_data

            print(filtered_bbox_data.keys())

            self.images = list(filtered_bbox_data.keys())
            self.bbox_list = self.get_bbox_list(filtered_bbox_data)
            self.total_img_count = len(filtered_bbox_data)
            print(len(filtered_bbox_data))
    

    def generate_ordering(self, lst):
        # sort unique elements and assign ranks
        sorted_vals = sorted(lst)
        val_to_rank = {val: rank + 1 for rank, val in enumerate(sorted_vals)}
        # map the original list to its ranks
        return [val_to_rank[x] for x in lst]
        
    def spearman(self, eyegaze_labels, human_ranks):
        max_spearman = 0.0
        for human_labels in human_ranks:
            # get common labels
            common_labels = list(set(human_labels) & set(eyegaze_labels))
                
                # assign ranks based on positions
            human_rankings = [human_labels.index(label) + 1 for label in common_labels]
            eyegaze_rankings = [eyegaze_labels.index(label) + 1 for label in common_labels]

            human_rankings = self.generate_ordering(human_rankings)
            eyegaze_rankings = self.generate_ordering(eyegaze_rankings)

            print(human_rankings)
            print(eyegaze_rankings)
                
                # calculate correlations
            if len(human_rankings) >= 2:  # need at least 2 to correlate
                spearman_corr, _ = spearmanr(human_rankings, eyegaze_rankings)
            else:
                if human_rankings[0] == eyegaze_rankings[0]:
                    spearman_corr = 1.0
            if spearman_corr > max_spearman:
                max_spearman = spearman_corr
        return max_spearman
      
    def set_rank_prompt_data(self, src_json, base_prompt_text, rank_data_dir=None):
        self.base_prompt_text = base_prompt_text
        with open(src_json, "r") as f:
            data = json.load(f)
        existing_ranks = {}
        if rank_data_dir is not None:
            with open(rank_data_dir, "r") as f:
                existing_ranks = json.load(f)
        data = {k: v for k, v in data.items() if existing_ranks.get(k) is None}
        self.images = list(data.keys())
        self.desc = list(data.values())

        print(len(self.images))
    
    def set_label_infuser_prompt_data(self, src_json, base_prompt_text, images_dir, metadata_save_dir, data_dir=None):
        self.base_prompt_text = base_prompt_text
        with open(src_json, "r") as f:
            data = json.load(f)
            print(len(data))
        with open(metadata_save_dir, "r") as f:
            metadata = json.load(f)
        existing_ranks = {}
        if data_dir is not None:
            with open(data_dir, "r") as f:
                existing_ranks = json.load(f)
        data = {k: v for k, v in data.items() if existing_ranks.get(k) is None}
        print(data.keys())

        self.metadata_save_dir = metadata_save_dir
        self.images_dir = images_dir
        # self.images = list(data.keys())
        # self.desc = list(data.values())
        self.bbox_list = self.get_label_infuser_bbox_list(data, metadata_save_dir=metadata_save_dir)


        print(len(self.images))
    

    def scramble_dict(self, d, seed=4):
        items = list(d.items())
        rng = random.Random(seed)
        rng.shuffle(items)
        return dict(items)
    
    def mask_to_coco_poly(self,binary_mask):
        # Convert mask to uint8
        mask = np.asfortranarray(binary_mask.astype(np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmentations = []
        for contour in contours:
            if len(contour) >= 6:  # 3 points minimum for a valid polygon
                segmentation = contour.flatten().tolist()
                segmentations.append(segmentation)
        return segmentations

    def get_bbox_list(self, data):
        bbox_list = []
        for img, objects in data.items():
            img_list = []
            print(f'{img}')
            for object_info in objects[:8]:
                if object_info['label'] is None:
                    continue
                img_list.append({
                    "label":object_info['label'],
                    "bbox":object_info['bbox']
                })
                print(object_info['label'])
            bbox_list.append(img_list)

        return bbox_list
    
    def get_label_infuser_bbox_list(self, data, metadata_save_dir):
        bbox_list = []
        object_data = PromptGenerator.read_json(metadata_save_dir)
        print(len(object_data))
        for idx, img in enumerate(object_data.keys()):
            if idx < 100:
                continue
            img_list = []
            print(f'{img}')
            objects = object_data[img]
            for object in objects:
                img_list.append({
                    "label":object['label'],
                    "bbox":object['bbox']
                })
                print(f'{object['label']}:{object['bbox']}')
            bbox_list.append(img_list)
            self.images.append(img)
            self.desc.append(data[os.path.splitext(img)[0]])
        # metadata = {}
        # model = maskrcnn_resnet50_fpn(pretrained=True)
        # model.eval()
        # available_images = os.listdir(self.images_dir)
        # for idx, img in enumerate(data.keys()):
        #     if len(bbox_list) > 300:
        #         break
        #     if img not in available_images:
        #         continue
        #     image_path = os.path.join(self.images_dir, img)
        #     image = Image.open(image_path).convert("RGB")
        #     img_tensor = F.to_tensor(image).unsqueeze(0)
        #     # Run prediction
        #     with torch.no_grad():
        #         predictions = model(img_tensor)

        #     # Extract results
        #     boxes = predictions[0]['boxes'].tolist()
        #     labels = predictions[0]['labels'].tolist()
        #     scores = predictions[0]['scores'].tolist()
        #     masks = predictions[0]['masks']  # [N, 1, H, W]

        #     img_data = []
        #     object_data = []
        #     counts = Counter(labels)

        #     # Confidence threshold (optional)
        #     threshold = 0.7
        #     keep = [i for i, score in enumerate(scores) if score > threshold]
        #     boxes = [boxes[i] for i in keep]
        #     labels_raw = [labels[i] for i in keep]
        #     masks = [masks[i][0].cpu().numpy() > 0.5 for i in keep]  # Threshold binary

        #     # Convert label IDs to names
        #     labels = [
        #         self.COCO_INSTANCE_CATEGORY_NAMES[label] if label < len(self.COCO_INSTANCE_CATEGORY_NAMES) else f"Label_{label}"
        #         for label in labels_raw
        #     ]

        #     # Only if one instance per class
        #     counts = Counter(labels)
        #     if all(count <= 1 for count in counts.values()):
        #         for label, box, mask in zip(labels, boxes, masks):
        #             x_min, y_min, x_max, y_max = box
        #             width = x_max - x_min
        #             height = y_max - y_min
        #             segmentation = self.mask_to_coco_poly(mask)
        #             object_data.append({
        #                 "label": label,
        #                 "bbox": [x_min, y_min, width, height],
        #                 "segmentation": segmentation
        #             })
        #             img_data.append({
        #                 "label": label,
        #                 "bbox": [x_min, y_min, width, height]
        #             })
        #             print(f'{label}: bbox={box}')
        #         bbox_list.append(img_data)
        #         self.images.append(img)
        #         self.desc.append(data[img])
        #         metadata[img] = object_data
        #         PromptGenerator.save_json(metadata_save_dir, metadata)


        return bbox_list
            


    def get_description_prompt(self):
        if self.index >= self.total_img_count:
            return None, None, None
        
        image = str(self.images[self.index])
        bbox_list = self.bbox_list[self.index]

        text = self.base_prompt_text + "\n" 
        for instance_info in bbox_list:
            text += (" " + instance_info["label"] + ":" + str(instance_info["bbox"]) + "\n")
        
        self.index += 1
        
        return text, image, self.index - 1 
    
    def get_description_prompt(self, object_data):
        prompt = self.base_prompt_text + "\n" 
        for instance_info in object_data.values():
            prompt += (" " + instance_info['object'].label + ":" + str(instance_info['object'].bbox) + "\n")
        
        return prompt

    def get_rank_prompt(self):
        if self.index >= len(self.images):
            return None, None, None
        
        image = str(self.images[self.index])
        desc = self.desc[self.index]

        text = self.base_prompt_text + desc
        
        self.index += 1
        
        return text, image, self.index - 1 
    
    def get_rank_prompt(self, desc):
        prompt = self.base_prompt_text + desc
        return prompt
    
    def get_label_infuser_prompt(self):
        if self.index >= len(self.images):
            return None, None, None
        
        image = str(self.images[self.index])
        desc = self.desc[self.index]
        bbox_list = self.bbox_list[self.index]

        text = self.base_prompt_text + desc + "\nBbox list: "
        for instance_info in bbox_list:
            text += (" " + instance_info["label"] + ":" + str(instance_info["bbox"]) + "\n")
        
        self.index += 1
        
        return text, image, self.index - 1 

    @staticmethod
    def read_json(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def save_json(json_file, json_list):
        with open(json_file, "r") as f:
            old_data = json.load(f)

        # Merge dict2 into dict1
        old_data.update(json_list)

        # Write the updated dictionary back to the same file
        with open(json_file, "w") as f:
            json.dump(old_data, f, indent=2)

