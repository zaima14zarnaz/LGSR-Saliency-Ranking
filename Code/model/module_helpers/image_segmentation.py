from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random
from pycocotools import mask as maskUtils


class ImageSegmentation:
    class SegmentedObject:
        def __init__(self, label, bbox, segmentations=None , category_name=None):
            self.label = label
            self.bbox = bbox
            self.category_name = category_name
            self.segmentations = segmentations
    
    def __init__(self, ann_file=None):
        if ann_file is None: 
            self.annotation_file = "/home/zaimaz/Desktop/research1/SaliencyRanking/Code/LGSR/anns/coco_anns/instances_train2014.json"
        else:
            self.annotation_file = ann_file
        self.coco = COCO(self.annotation_file)
        self.annotations = {}
        self.image_info = {}

    def get_image_id_from_filename(self, filename):
        """
        Get the image ID from the COCO dataset using the filename.
        """
        image_ids = self.coco.getImgIds()
        images = self.coco.loadImgs(image_ids)

        for img in images:
            if img['file_name'] == filename:
                return img['id'], img
        
        return None, None  # Return None if the filename is not found
    
    def draw_segmentations(self, image_info):
        """
        Get the segmentation mask for a given COCO image.
        """
        if 'image_path' not in image_info:
            print("Error: 'image_path' is missing in image_info.")
            return None, None, None

        image_path = image_info['image_path']
        
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None, None, None

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: OpenCV failed to read image {image_path}")
            return None, None, None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get annotation IDs
        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        annotations = self.coco.loadAnns(ann_ids)

        # Create segmentation mask
        height, width = image_info['height'], image_info['width']
        segmentation_mask = np.zeros((height, width), dtype=np.uint8)

        for ann in self.annotations:
            for seg in ann['segmentation']:
                polygon = np.array(seg).reshape((len(seg) // 2, 2))
                cv2.fillPoly(segmentation_mask, [polygon.astype(np.int32)], 255)

        return image, segmentation_mask, self.annotations
    
    def apply_transparent_mask(self, image, mask, alpha=0.5):
        """
        Apply a semi-transparent segmentation mask overlay and add labels.
        """
        print(f"Applying mask for {len(self.annotations)} objects")

        # Convert grayscale mask to 3-channel color
        mask_colored = np.zeros_like(image, dtype=np.uint8)
        mask_colored[mask > 0] = (255, 0, 0)  # Red mask

        # Blend images using alpha transparency
        overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)

        # Add labels on objects
        for ann in self.annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            category_id = ann['category_id']
            category_name = self.coco.loadCats(category_id)[0]['name']
            # print(f'Overlaying on {category_name}:{bbox}')

            # Calculate text position
            x, y = int(bbox[0]), int(bbox[1] - 10)  # Position label above the object
            text_size = 1  # Text scale
            text_color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)  # Black background for text
            
            # Draw rectangle background for label
            (w, h), _ = cv2.getTextSize(category_name, cv2.FONT_HERSHEY_SIMPLEX, text_size, 2)
            cv2.rectangle(overlay, (x, y - h - 5), (x + w, y + 5), bg_color, -1)
            
            # Put label text on the image
            cv2.putText(overlay, category_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, text_size, text_color, 2, cv2.LINE_AA)

        return overlay

    def save_segmented_image(self, save_path, segmented_image):
        """
        Save the segmented image.
        """
        if segmented_image is None:
            print("Error: Cannot save NoneType image.")
            return

        cv2.imwrite(save_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
        print(f"Segmented image saved to: {save_path}")

    def get_segmentations(self, filename, image_path, save_path=None):
        # print("In get segmentations")
        image_id, image_info = self.get_image_id_from_filename(filename)
        if image_id is None:
            print(f"Error: Image '{filename}' not found in COCO dataset.")
            return False

        # print(f"Image ID: {image_id}")

        image_info['image_path'] = image_path
        image, segmentation_mask, self.annotations = self.draw_segmentations(image_info)

        if self.annotations is None or len(self.annotations) <= 0:
            self.annotations = self.get_annotations(filename=filename)
        
        if image is None or segmentation_mask is None:
            print("Error: Segmentation failed. Cannot save.")
            return False

        # Apply semi-transparent segmentation mask and add labels

        if save_path is not None:
            # Save segmented image
            overlay = self.apply_transparent_mask(image, segmentation_mask, alpha=0.5)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.save_segmented_image(save_path, overlay)
        
        return True
    
    def load_annotations(self, image_list):
        self.coco = COCO(self.annotation_file)
        self.annotations = {}
        self.image_info = {}

        # Build mapping from filename to full image dict
        fname_to_img = {img['file_name']: img for img in self.coco.dataset['images']}

        for fname in image_list:
            img = fname_to_img.get(fname)
            if img is not None:
                image_id = img['id']
                ann_ids = self.coco.getAnnIds(imgIds=image_id)
                anns = self.coco.loadAnns(ann_ids)
                self.annotations[fname] = anns
                self.image_info[fname] = {
                    'width': img['width'],
                    'height': img['height'],
                    'id': image_id
                }
            else:
                self.annotations[fname] = None
                self.image_info[fname] = None

    
    def get_annotations(self, filename):
        if filename in self.annotations.keys():
            return self.annotations[filename]
        return None
    
    @staticmethod
    def get_suffix(label):
        suffix_start_pos = label.rfind('_')
        if suffix_start_pos == -1:
            return -1
        suffix_str = label[suffix_start_pos+1:]
        suffix_id = int(suffix_str)

        return suffix_id

    
    def add_obj(self, obj, instances):
        category_name = obj.category_name
        if category_name not in instances.keys():
            instances[category_name] = {}
            obj.label = category_name+'_1'
            instances[category_name][obj] = 0
            return
        
        for obj_i in instances[category_name].keys():
            if obj.bbox == obj_i.bbox and obj.category_name==obj_i.category_name:
                instances[category_name][obj_i] = instances[category_name][obj_i] + 1
                return 

        last_object, last_score = next(reversed(instances[category_name].items()))
        suffix_id = ImageSegmentation.get_suffix(last_object.label)
        if suffix_id == -1:
            print("Invalid object label, cannot find suffix")
            return
        obj.label = category_name + "_" + str(suffix_id+1)
        instances[category_name][obj] = 0

    
    def label_instances(self, filename):
        instances = {}
        annotations = self.get_annotations(filename=filename)
        if annotations == None:
            return None, None, None
        for ann in annotations:
            category_id = ann['category_id']
            category_name = self.coco.loadCats(category_id)[0]['name']
            bbox = ann['bbox']
            seg = ann['segmentation']
            obj = ImageSegmentation.SegmentedObject(category_name, bbox, seg, category_name)
            self.add_obj(obj, instances)
        instances['bg'] = {}
        instances['bg'][ImageSegmentation.SegmentedObject('bg',[0,0,0,0], None, 'bg')] = 0
        
        obj_list = {}
        for category_name, instance_info in instances.items():
            for obj, _ in instance_info.items():
                obj_list[obj.label] = obj
        return instances, obj_list, annotations

    def get_img_width_height(self, img_fname):
        return self.image_info[img_fname]['width'], self.image_info[img_fname]['height']
    
    def get_object_pixels(self, segmentation, height, width):
        """
        Returns a list of (x, y) pixel coordinates inside the object mask.
        """
        if isinstance(segmentation, list):
            rles = maskUtils.frPyObjects(segmentation, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segmentation, dict) and 'counts' in segmentation:
            rle = segmentation
        else:
            raise ValueError("Unknown segmentation format")

        mask = maskUtils.decode(rle)
        pixels = np.argwhere(mask == 1)
        return [[int(x), int(y)] for y, x in pixels]  # Convert to native int



