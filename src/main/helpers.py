import os
import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from Point2D import Point2D
from Point3D import Point3D


class helpers:
    def __init__(self):
        pass
    
    def loadImages(self, img_dir, should_resize):
        "return list of numpy array of images"
        imagesList = []
        valid_images = (".jpg",".gif",".png",".tga")
        
        index = 0
        for file in sorted(os.listdir(img_dir)):
            if file.endswith(valid_images):
                imagesList.append(file)    
            index +=1

        loadedImages = []
        for image in imagesList:
            img = cv2.imread(img_dir + image)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if should_resize:
                img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5 )
            # img = np.float32(img)
            loadedImages.append(img)

        return loadedImages
    
    def load_poses(self, K_file):
        "Load poses K from K file"

        with open(K_file) as f:
            lines = f.readlines()     
            lines = [line.strip(", \n") for line in lines]
            K = np.genfromtxt(lines, dtype = float, delimiter = ", ")
        return K
        
    def Point2DListToInt(self, keypoints):
        """
        keypoints: List of Point2D objects
        out: List of [x,y] keypoints in integers
        """
        return np.array([[k.u,k.v] for k in keypoints])
    
    def IntListToPoint2D(self, keypoints):
        """
        keypoints: List of [x,y] keypoints in integers
        out: List of Point2D objects
        """
        return [Point2D(keypoints[0][idx], keypoints[1][idx]) for idx in range(0, len(keypoints[0]))]
    
    def IntListToPoint3D(self, landmarks):
        """
        keypoints: List of [X, Y, Z] landmarks in integers
        out: List of Point3D objects
        """
        return [Point3D(landmarks[0][idx], landmarks[1][idx], landmarks[2][idx]) for idx in range(0, len(landmarks[0]))]

    def IntListto3D(self, landmarks):
        """
        in: List of int landmarks
        out: list of [X, Y, Z] landmarks
        """
        landmarks = landmarks[:3]
        new_landmarks = [[landmarks[0][i], landmarks[1][i], landmarks[2][i]] for i in range(len(landmarks[0]))]
        return np.array(new_landmarks)
    
    def describe_keypoints(self, r, image, keypoints):
        # out: return the List of keypoint descriptors
        size = 2*r + 1
        keypoints_decription = np.zeros((len(keypoints), size*size))
        temp_img = np.pad(image, [(r, r), (r,r)], mode='constant')
        
        for idx, kpt in enumerate(keypoints):
            patch = temp_img[kpt.u:kpt.u + size, kpt.v:kpt.v + size].flatten()
            keypoints_decription[idx,:] = patch

        keypoints_decription = np.array(keypoints_decription)
        return keypoints_decription
    def kpts2kpts2Object(self, kpts):
        return list(cv2.KeyPoint(kpts[i][0], kpts[i][1],2 ) for i in range(len(kpts)))
    
    def generate_trajectory(self, points):
        fig, ax = plt.subplots()
        xdata, ydata = [], []
        ln, = plt.plot([], [], 'ro')

        def init():
            ax.set_xlim(-100, 100)
            ax.set_ylim(0, 200)
            return ln,

        def update(points):
            xdata.append(points[0])
            ydata.append(points[1])
            ln.set_data(xdata, ydata)
            return ln,

        ani = FuncAnimation(fig, update, frames=points,
                            init_func=init, blit=True)
        plt.show()

    def read_yaml(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)