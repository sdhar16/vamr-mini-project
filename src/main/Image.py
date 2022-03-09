import cv2 
import numpy as np 
from Point2D import Point2D

class Image:
    def __init__(self, image):
        self.image = image

        self.generate_harris_score()
        self.select_keypoints(300, 8)
        self.describe_keypoints(8)
    
    def generate_harris_score(self):
        # TODO: Tune parameters of cornerHarris funtion from opencv
        self.harris_score = cv2.cornerHarris(self.image,2,3,0.04)

        self.harris_score = cv2.dilate(self.harris_score,None)

        # TODO: Tune parameters
        ret, self.harris_score = cv2.threshold(self.harris_score,0.01*self.harris_score.max(),255,0)

        self.harris_score = np.uint8(self.harris_score)

    def select_keypoints(self, num, r):
        # input number of selected keypoints and patch radius
        self.keypoints = []
        temp_scores = np.pad(self.harris_score,[(r,r),(r,r)],mode = 'constant')        

        for i in range(num):
            max_index = np.array(np.unravel_index(temp_scores.argmax(), temp_scores.shape))
            if(temp_scores[max_index[0], max_index[1]]<=0):  
                # Break if max keypoint's score get to zero or below zero.
                break
            diff = max_index-r
            self.keypoints.append(Point2D(int(diff[0]), int(diff[1])))
            temp_scores[max_index[0]-r:max_index[0]+r+1,max_index[1]-r:max_index[1]+r+1] = 0

    def describe_keypoints(self, r):
        # out: return the List of keypoint descriptors
        size = 2*r + 1
        self.keypoints_decription = np.zeros((len(self.keypoints), size*size))
        temp_img = np.pad(self.image, [(r, r), (r,r)], mode='constant')
        
        for idx, kpt in enumerate(self.keypoints):
            patch = temp_img[kpt.u:kpt.u + size, kpt.v:kpt.v + size].flatten()
            self.keypoints_decription[idx,:] = patch

        self.keypoints_description = np.array(self.keypoints_decription)

    def get_keypoints(self):
        return self.keypoints.copy()
    
    def get_keypoints_descriptions(self):
        return self.keypoints_decription.copy()

    
