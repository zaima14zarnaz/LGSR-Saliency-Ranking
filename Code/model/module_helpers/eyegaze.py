import cv2
import numpy as np
import random

from Code.model.module_helpers.fixationlocs import FixationLocs
from Code.model.module_helpers.image_segmentation import ImageSegmentation

import torch
import numpy as np
import cv2
from collections import defaultdict

class EyeGazeAnalyzer:
    GAZE_INC = 1
    FIXATION_INC = 2
    TIME_DECAY = 0.001
    REPETITION_BOOST = 0.01

    BACKGROUND_OBJECT = ImageSegmentation.SegmentedObject('bg', [0,0,0,0], None, 'bg')

    def __init__(self, segmentor, device='cuda'):
        self.segmentor = segmentor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.object_map = {}  # category to object map
        self.object_scores = {}  # SegmentedObject to score

    def set_data(self, img_fname, img_addr, fixation_addr, annotations):
        self.annotations = annotations
        self.img_fname = img_fname
        self.img_addr = img_addr
        self.data_fname = fixation_addr
        self.object_map = {} 
        self.object_scores = {}  

        
        self.repeats = defaultdict(int)
        self.previous_object = None

    def rank_objects(self, instances):
        self.object_map = instances
        all_gaze_coors, all_fixation_locs = self.get_eye_gaze_coordinates()

        self.find_object_scores(all_gaze_coors, EyeGazeAnalyzer.GAZE_INC)
        self.find_object_scores(all_fixation_locs, EyeGazeAnalyzer.FIXATION_INC)

        return self.object_map, self.object_scores

    def find_object_scores(self, gaze_coors, inc_amount):
        for candidate_gaze_list in gaze_coors:
            coors = torch.tensor(candidate_gaze_list, dtype=torch.float32, device=self.device)
            for i in range(coors.size(0)):
                x, y = int(coors[i][0].item()), int(coors[i][1].item())
                obj = self.get_object(x, y)
                if self.is_same_as_previous(obj):
                    self.repeats[obj] += 1
                self.add_obj(obj, inc_amount, i, self.repeats[obj])
                self.set_previous_object(obj)

    def get_eye_gaze_coordinates(self, data_fname=None, fixationLocs=None):
        if data_fname is not None:
            self.data_fname = data_fname
        fixation_locs = FixationLocs()
        all_gaze_coors, all_fixation_locs = fixation_locs.extract_salicon_locs(self.data_fname)
        return all_gaze_coors, all_fixation_locs

    def add_obj(self, obj, gaze_idx, inc_amount, repeats=None):
        label, bbox, category_name = obj.label, obj.bbox, obj.category_name
        time_decay = EyeGazeAnalyzer.TIME_DECAY
        repetition_boost = EyeGazeAnalyzer.REPETITION_BOOST

        if category_name not in self.object_map:
            self.object_map[category_name] = {}

        if category_name == 'bg':
            if self.object_map[category_name]:
                first_key = next(iter(self.object_map[category_name]))
                score_tensor = torch.tensor([self.object_map[category_name][first_key]], device=self.device)
                update = torch.tensor([inc_amount - gaze_idx*time_decay], device=self.device)
                new_score = (score_tensor + update).item()
                self.object_map[category_name][first_key] = round(new_score, 2)
                self.object_scores[first_key] = round(new_score, 2)
                return
            else:
                self.object_map[category_name][obj] = 1
                self.object_scores[obj] = 1
                return

        for obj_i in self.object_map[category_name].keys():
            if obj.bbox == obj_i.bbox and obj.category_name == obj_i.category_name:
                base_score = torch.tensor([self.object_map[category_name][obj_i]], device=self.device)
                boost = torch.tensor([inc_amount - time_decay], device=self.device)
                new_score = (base_score + boost).item()
                self.object_map[category_name][obj_i] = round(new_score, 2)
                self.object_scores[obj_i] = round(new_score, 2)
                return

        self.object_map[category_name][obj] = 1
        self.object_scores[obj] = 1

    def get_object(self, x, y):
        for objs in self.object_map.values():
            for obj in objs.keys():
                if obj.segmentations is not None:
                    for seg in obj.segmentations:
                        if not seg or isinstance(seg, str): 
                            continue
                        polygon = np.array(seg).flatten().reshape(-1, 2).astype(np.float32)
                        if len(polygon) < 3:
                            continue
                        if cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
                            return obj
        return EyeGazeAnalyzer.BACKGROUND_OBJECT

    def is_same_as_previous(self, current_object):
        return self.previous_object == current_object

    def set_previous_object(self, current_object):
        if current_object.label != EyeGazeAnalyzer.BACKGROUND_OBJECT.label:
            self.previous_object = current_object


# import cv2
# import numpy as np
# from fixationlocs import FixationLocs
# from segmentation.image_segmentation import ImageSegmentation
# from segmented_object import SegmentedObject

# class EyeGazeAnalyzer:
#     GAZE_INC = 1
#     FIXATION_INC = 2
#     TIME_DECAY = 0.001
#     REPETITION_BOOST = 1.1
#     BACKGROUND_OBJECT = SegmentedObject('bg', [0, 0, 0, 0], None, 'bg')

#     def __init__(self, img_fname, data_fname, annotations, segmentor):
#         self.annotations = annotations
#         self.img_fname = img_fname
#         self.data_fname = data_fname
#         self.segmentor = segmentor
#         self.object_map = {}
#         self.repeats = {}
#         self.previous_object = None

#     def sort_objects(self):
#         self.object_map = self.segmentor.label_instances(self.img_fname)
#         all_gaze_coors, all_fixation_locs = self.get_eye_gaze_coordinates()

#         for coords, inc in zip([all_gaze_coors, all_fixation_locs], [self.GAZE_INC, self.FIXATION_INC]):
#             self.find_object_scores(coords, inc, self.TIME_DECAY, self.REPETITION_BOOST)

#         return self.object_map

#     def find_object_scores(self, gaze_coors, inc_amount, time_decay, repetition_boost):
#         for candidate_gaze_list in gaze_coors:
#             coords = np.round(candidate_gaze_list).astype(int)
#             for x, y in coords:
#                 obj = self.get_object(x, y)
#                 if self.is_same_as_previous(obj):
#                     self.repeats[obj] = self.repeats.get(obj, -1) + 1
#                 self.add_obj(obj, inc_amount, time_decay, repetition_boost, self.repeats.get(obj, 0))
#                 self.set_previous_object(obj)

#     def get_eye_gaze_coordinates(self):
#         return FixationLocs().extract_salicon_locs(self.data_fname)

#     def add_obj(self, obj, inc_amount, time_decay, repetition_boost, repeats):
#         category_name = obj.category_name
#         self.object_map.setdefault(category_name, {})

#         if category_name == 'bg':
#             if self.object_map[category_name]:
#                 first_key = next(iter(self.object_map[category_name]))
#                 self.object_map[category_name][first_key] += inc_amount - time_decay
#             else:
#                 self.object_map[category_name][obj] = 1
#             return

#         for obj_i in self.object_map[category_name]:
#             if obj.bbox == obj_i.bbox and obj.category_name == obj_i.category_name:
#                 self.object_map[category_name][obj_i] += inc_amount - time_decay + (repeats * repetition_boost)
#                 return

#         self.object_map[category_name][obj] = 1

#     def get_object(self, x, y):
#         for objs in self.object_map.values():
#             for obj in objs:
#                 if obj.segmentations:
#                     for seg in obj.segmentations:
#                         if seg:
#                             polygon = np.array(seg, dtype=np.float32).reshape(-1, 2)
#                             if len(polygon) >= 3 and cv2.pointPolygonTest(polygon, (x, y), False) >= 0:
#                                 return obj
#         return self.BACKGROUND_OBJECT

#     def is_same_as_previous(self, current_object):
#         return self.previous_object != current_object

#     def set_previous_object(self, current_object):
#         if current_object.label != self.BACKGROUND_OBJECT.label:
#             self.previous_object = current_object

#     def extract_objects(self, annotations):
#         self.object_map.clear()
#         for ann in annotations:
#             category_id = ann['category_id']
#             category_name = self.segmentor.coco.loadCats(category_id)[0]['name']
#             bbox = ann['bbox']
#             self.object_map.setdefault(category_name, {})



# segmentor = ImageSegmentation()
# image_filename = "COCO_train2014_000000567976.jpg"
# data_fname = "COCO_train2014_000000567976.mat"
# image_path = "/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/eyegaze/COCO_train2014_000000567976.jpg"
# annotations = segmentor.get_annotations(image_filename)

# eyegaze_analyzer = EyeGazeAnalyzer(annotations=annotations, img_fname=image_filename, data_fname=data_fname, segmentor=segmentor)
# sorted_objects = eyegaze_analyzer.sort_objects()

# for category_name, objs in sorted_objects.items():
#     print(category_name+":")
#     for obj, score in objs.items():
#         print(f'{obj.label} ({obj.bbox}) = {round(score,2)}')
