import cv2
import numpy as np
from helpers import helpers

class Initialization:
    def __init__(self, img1, img2, K, config):
        self.image1 = img1
        self.image2 = img2
        self.K = K
        self.config = config
        self.helpers = helpers()

    def run(self):
        kpts1, kpts2 = self.klt_matching(self.image1, self.image2)
        E, inliers1, inliers2 = self.getEssentialMatrix(kpts1, kpts2)  
        landmarks, R, T = self.disambiguateEssential(E, inliers1, inliers2)  
        T = -R @ T
        
        # print(inliers2.shape)
        # print(landmarks.shape)
        
        landmarks = landmarks.T
        inliers2 = inliers2.T
        
        idx = np.where(landmarks[:,2]>0)
        landmarks = landmarks[idx]
        inliers2 = inliers2[idx]
        
        
        inliers2 = inliers2.T
        landmarks = landmarks.T
        
        inliers2 = self.helpers.IntListToPoint2D(inliers2)
        landmarks = self.helpers.IntListto3D(landmarks)
        return inliers2,landmarks, T


    def klt_matching(self, image1, image2):
        image1 = np.uint8(image1)
        image2 = np.uint8(image2)

        feature_params = dict(maxCorners = self.config["ShiTomasi_params"]["maxCorners"],
                            qualityLevel = self.config["ShiTomasi_params"]["qualityLevel"],
                            minDistance = self.config["ShiTomasi_params"]["minDistance"],
                            blockSize = self.config["ShiTomasi_params"]["blockSize"])
        
        lk_params = dict( winSize  = (self.config["KLT_params"]["winSize"][0],self.config["KLT_params"]["winSize"][1]),
                  maxLevel = self.config["KLT_params"]["maxLevel"],
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.config["KLT_params"]["EPS"], self.config["KLT_params"]["COUNT"]))

        p0 = cv2.goodFeaturesToTrack(image1, mask = None, **feature_params)

        p1, st, _ = cv2.calcOpticalFlowPyrLK(image1, image2, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        return good_old, good_new

    def disambiguateEssential(self, E, inliers1, inliers2):
        """
        out: return point3D, R, T
        """
        R1, R2, T = cv2.decomposeEssentialMat(E) 
        
        # T = T *0.1
        
        rvec1,_ = cv2.Rodrigues(R1)
        rvec2,_ = cv2.Rodrigues(R2)
        
        if np.linalg.norm(rvec1)>1.5:
            R1 = R2
        elif np.linalg.norm(rvec2)>1.5:
            R2 = R1
        
        # points3D_1, sum_left_1, sum_right_1 = self.triangulate(R1.T, -R1.T @ T, inliers1, inliers2)
        # points3D_2, sum_left_2, sum_right_2 = self.triangulate(R2.T, -R2.T @ T, inliers1, inliers2)
        # points3D_3, sum_left_3, sum_right_3 = self.triangulate(R1.T, -R1.T @ -T, inliers1, inliers2)
        # points3D_4, sum_left_4, sum_right_4 = self.triangulate(R2.T, -R2.T @ -T, inliers1, inliers2)
        
        
        points3D_1, sum_left_1, sum_right_1 = self.triangulate(R1,  T, inliers1, inliers2)
        points3D_2, sum_left_2, sum_right_2 = self.triangulate(R2, T, inliers1, inliers2)
        points3D_3, sum_left_3, sum_right_3 = self.triangulate(R1, -T, inliers1, inliers2)
        points3D_4, sum_left_4, sum_right_4 = self.triangulate(R2, -T, inliers1, inliers2)


        max_points = 0
        if sum_left_1 == sum_right_1:
            point3D, R, T = points3D_1.copy(), R1.copy(), T.copy()
            max_points = sum_left_1
        if sum_left_2 == sum_right_2 and sum_left_2 >= max_points:
            point3D, R, T  = points3D_2.copy(), R2.copy(), T.copy()
            max_points = sum_left_2
        if sum_left_3 == sum_right_3 and sum_left_3 >= max_points:
            point3D, R, T  = points3D_3.copy(), R1.copy(), -T.copy()
            max_points = sum_left_3
        if sum_left_4 == sum_right_4 and sum_left_4 >= max_points:
            point3D, R, T  = points3D_4.copy(), R2.copy(), -T.copy()
            max_points = sum_left_4

        return point3D, R, T
    
    def triangulate(self, R,t, inliers1, inliers2):
        """
        out: return 3D points, sum of relative coordinates of left image if z greater than 0, 
        sum of relative coordinates of right image if z greater than 0
        """
        inliers1 = np.vstack((inliers1, np.ones(inliers1.shape[1])))
        inliers2 = np.vstack((inliers2, np.ones(inliers1.shape[1])))

        norm_inliers1 = np.linalg.inv(self.K) @ inliers1
        norm_inliers2 = np.linalg.inv(self.K) @ inliers2

        identity = cv2.hconcat([np.eye(3), np.zeros((3,1))])
        M1 = np.concatenate((R, t), axis=1) 

        points3D = cv2.triangulatePoints(identity, M1, norm_inliers1[:2,:], norm_inliers2[:2,:])
        points3D /= points3D[3]
        
        relative_p1 = identity @ points3D
        relative_p2 = M1 @  points3D

        return (points3D, sum(relative_p1[2]>0), sum(relative_p2[2]>0))
    
    def getEssentialMatrix(self, kpts1, kpts2):
        """
        kpt_matching: keypoints matching between 2 images [[Point2D keypoint 1, Point2D keypoint 2]]
        """
        F, mask = cv2.findFundamentalMat(kpts1, kpts2, cv2.FM_RANSAC)

        inliers1 = np.array([[kpts1[i][0], kpts1[i][1]] for i,j in enumerate(mask) if j[0] == 1]).T
        inliers2 = np.array([[kpts2[i][0], kpts2[i][1]] for i,j in enumerate(mask) if j[0] == 1]).T
        
        return self.K.T @ F @ self.K, inliers1, inliers2
