import scipy.io
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt



class FixationLocs:
    @staticmethod
    def extract_from_mat(filename, loc_column_name):
        mat_data = scipy.io.loadmat(file_name=filename)
        fix_locs = np.array(mat_data[loc_column_name])
        y, x = np.nonzero(fix_locs)
        coor = list(zip(x,y))
        coor.sort()

        return coor
    
    @staticmethod
    def extract_from_maps(filename, threshold):
        fixation_map = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        y_coords, x_coords = np.where(fixation_map > threshold)
        coor = list(zip(x_coords, y_coords))
        coor.sort()
        
        return coor
    
    @staticmethod
    def extract_CAT2000_locs(filename):
        variable_name = 'fixLocs'
        return FixationLocs.extract_from_mat(filename, variable_name)
    
    @staticmethod
    def extract_figrim_locs(filename):
        variable_name = 'fixLocs'
        return FixationLocs.extract_from_mat(filename, variable_name)
    
    @staticmethod
    def extract_mit_locs(filename, threshold=50):
        return FixationLocs.extract_from_maps(filename, threshold)
    
    @staticmethod
    def extract_salicon_locs(data_fname, image_fname=None):
        if image_fname is not None:
            image = cv2.imread(image_fname)
        else:
            image = None
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        mat_data = scipy.io.loadmat(file_name=data_fname)
        gaze_col = "gaze"
        gaze_data = mat_data[gaze_col]


        # Display using Matplotlib
        # plt.imshow(image)
        # plt.axis("off")  # Hide axes
        # plt.show()
        radius = 1  # Circle radius
        gaze_color = (0, 0, 255)  # BGR (Red)
        fixation_color = (255, 0, 0)
        thickness = -1  # Negative thickness fills the circle
        all_gaze_coors = []
        all_fixation_locs = []
        for candidate_no in range(0,len(gaze_data)):
            gaze_coors = gaze_data[candidate_no][0][0]
            timestamps = gaze_data[candidate_no][0][1]
            fixation_locs = gaze_data[candidate_no][0][2]
            all_gaze_coors.append(gaze_coors)
            all_fixation_locs.append(fixation_locs)
            if image is not None: 
                for coor in gaze_coors:
                    x = int(coor[0])
                    y = int(coor[1])
                    cv2.circle(image, (x,y), radius, gaze_color, thickness)
                for coor in fixation_locs:
                    x = coor[0]
                    y = coor[1]
                    cv2.circle(image, (x,y), radius+2, fixation_color, thickness)
        # if image is not None:
            # cv2.imwrite("/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/eyegaze/image.jpg", image)
        return all_gaze_coors, all_fixation_locs








# mat_locs_extractor = FixationLocs()
# filename_cat2000 = "/home/zaimaz/Desktop/research1/SaliencyRanking/Dataset/EyegazeDatasets/CAT2000/trainSet/FIXATIONLOCS/Action/001.mat"
# filename_figrim = "/home/zaimaz/Desktop/research1/SaliencyRanking/Dataset/EyegazeDatasets/FIGRIM/Fillers/FIXATIONLOCS_fillers/airport_terminal/sun_aaajuldidvlcyzhv.mat"
# filename_mit = "/home/zaimaz/Desktop/research1/SaliencyRanking/Dataset/EyegazeDatasets/MIT/ALLFIXATIONMAPS/i05june05_static_street_boston_p1010764_fixMap.jpg"
# image_fname_salicon = "/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/eyegaze/COCO_train2014_000000567976.jpg"
# data_fname_salicon = "/home/zaimaz/Desktop/research1/SaliencyRanking/Code/groundTruth/eyegaze/COCO_train2014_000000567976.mat"

 
# # coor = mat_locs_extractor.extract_CAT2000_locs(filename=filename_cat2000)
# # print(len(coor))
# # coor = mat_locs_extractor.extract_figrim_locs(filename_figrim)
# # print(len(coor))
# # coor = mat_locs_extractor.extract_mit_locs(filename=filename_mit, threshold=230)
# # print(len(coor))
# mat_locs_extractor.extract_salicon_locs(image_fname=image_fname_salicon, data_fname=data_fname_salicon)

# # for x,y in coor:
# #     print(f'({x},{y})')



        



