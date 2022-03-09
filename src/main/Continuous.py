import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
from helpers import helpers

class Continuous:
    def __init__(self, keypoints, landmarks, T, images, K, config, baseline):
        self.h = helpers()
        self.init_T = T
        self.K = K
        self.init_keypoints = keypoints
        self.init_landmarks = landmarks
        self.images = list(map(np.uint8, images))
        self.config = config
        self.baseline = baseline
        self.lk_params = dict( winSize  = (self.config["KLT_params"]["winSize"][0],self.config["KLT_params"]["winSize"][1]),
                  maxLevel = self.config["KLT_params"]["maxLevel"],
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                  self.config["KLT_params"]["EPS"], 
                  self.config["KLT_params"]["COUNT"]))
    
    def run(self):
        T_X = [self.init_T[0][0]]
        T_Z = [self.init_T[2][0]]

        fig, ax = plt.subplots(2,2)

        p0 = self.h.Point2DListToInt(self.init_keypoints)
        p0 = np.float32(p0.reshape(-1, 1, 2))

        good_img_landmarks = self.init_landmarks
        ax[1,1].axis(xmin=T_X[-1]-2, xmax=T_X[-1]+2, ymin=T_Z[-1]-2, ymax=T_Z[-1]+2)

        for i in range(0, min(len(self.images),100000)):
            ax[0,1].axis(xmin=min(T_X) - 1, xmax=max(T_X) + 1, ymin =min(T_Z) - 1, ymax=max(T_Z) + 1)

            if i<=self.baseline[1]:
                continue
            
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.images[i-1], self.images[i], p0, None, **self.lk_params)

            if p1 is not None:
                good_img_keypoints = p1[st1==1]
                temp_lst= []

                for index, value in enumerate(st1):
                    if(value==1):
                        temp_lst.append(good_img_landmarks[index])
                    
                good_img_landmarks = np.array(temp_lst)

            if len(good_img_keypoints) > 4:
                retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    good_img_landmarks, good_img_keypoints, self.K, None, flags=cv2.SOLVEPNP_P3P, confidence=0.9999
                )

                inliers = np.squeeze(np.array(inliers))
                good_img_keypoints_temp = good_img_keypoints
                good_img_keypoints = good_img_keypoints[inliers,:]
                good_img_landmarks = good_img_landmarks[inliers,:]
                
                # inverse tvec and rvec
                R,_ = cv2.Rodrigues(rvec)
                R = R.T
                rvec,_ = cv2.Rodrigues(R)        
                tvec = -R @ tvec
                
                if tvec[1] > 1:
                    tvec[1] = 1
                if tvec[1] < -1:
                    tvec[1] = -1    

                T_X.append(tvec[0])
                T_Z.append(tvec[2])
        
            # logic to add candidate keypoints
            
            # First: calculate new keypoints using ShiTomasi
            feature_params = dict(maxCorners = self.config["ShiTomasi_params"]["maxCorners"],
                            qualityLevel = self.config["ShiTomasi_params"]["qualityLevel"],
                            minDistance = self.config["ShiTomasi_params"]["minDistance"],
                            blockSize = self.config["ShiTomasi_params"]["blockSize"])

            img_kpts = cv2.goodFeaturesToTrack(self.images[i], mask = None, **feature_params)
            img_kpts = np.squeeze(img_kpts)

            # Second: img_kpts which are far away from good_img_keypoints are possible candidates kpts
            k = 0
            
            new_candidate = np.zeros((0,2)).astype('float32')
            for idx in range(img_kpts.shape[0]):
                a = img_kpts[idx,:]
                min_norm = 1e6
                for j in range(good_img_keypoints_temp.shape[0]):
 
                    b = good_img_keypoints_temp[j,:]
                    norm = np.linalg.norm(a-b)
                    if min_norm>norm:
                        min_norm = norm

                # this norm can be tuned
                if min_norm > 10:
                    if k == 0:
                        new_candidate = a
                    else:
                        new_candidate = np.vstack([new_candidate,a])
                    k = k+1

            if i == self.baseline[1]+1:
                candidate_kpts = new_candidate
                rvec_candidate = np.zeros([new_candidate.shape[0],3])
                tvec_candidate = np.zeros([new_candidate.shape[0],3])
                for j in range(new_candidate.shape[0]):
                    rvec_candidate[j,:] = rvec.T
                    tvec_candidate[j,:] = tvec.T

                fir_obs_C = candidate_kpts
                
            elif(candidate_kpts.shape[0]>0):
                # Third: candidate_kpts is good if it is tracked in next frame
                good_candidate_kpts, st, _ = cv2.calcOpticalFlowPyrLK(
                    self.images[i-1], self.images[i], candidate_kpts, None, **self.lk_params
                )

                if good_candidate_kpts is not None:
                    good_candidate_kpts = good_candidate_kpts[st==1]

                    # get rvec_candidate and fir_obs_C based on succesful tracking of candidate kpts
                    temp_rvec= []
                    temp_tvec = []
                    temp_fir_obs_C = []

                    for index, value in enumerate(st):
                        if(value==1):
                            temp_rvec.append(rvec_candidate[index])
                            temp_tvec.append(tvec_candidate[index])
                            temp_fir_obs_C.append(fir_obs_C[index])


                    rvec_candidate = np.array(temp_rvec)
                    tvec_candidate = np.array(temp_tvec)
                    fir_obs_C = temp_fir_obs_C

                # Fourth: Find completely new candidate kpts which are never found is previous frames
                # So new_candidate kpts which are far from good_candidate_kpts are completely new candidate kpts.
                k = 0

                selected_new_candidate = np.zeros((0,2)).astype('float32')
                for idx in range(new_candidate.shape[0]):
                    if(len(new_candidate.shape)!=2):
                        break

                    a = new_candidate[idx,:]
                    min_norm = 10000000
                    for j in range(candidate_kpts.shape[0]):
 
                        b = candidate_kpts[j,:]
                        norm = np.linalg.norm(a-b)
                        if min_norm>norm:
                            min_norm = norm
                            
                    # this norm can be tuned too
                    if min_norm > 10:
                        if k == 0:
                            selected_new_candidate = a
                        else:
                            selected_new_candidate = np.vstack([selected_new_candidate,a])
                        k = k+1

                # getting rvec and tvec of completely new candidates keypoints
                new_candidate_rvec = np.zeros([selected_new_candidate.shape[0],3])
                new_candidate_tvec = np.zeros([selected_new_candidate.shape[0],3])
                for l in range(selected_new_candidate.shape[0]):
                    new_candidate_rvec[l,:] = rvec.T
                    new_candidate_tvec[l,:] = tvec.T

                #Fifth: stacking old with newly created keypoints
                candidate_kpts = np.vstack([good_candidate_kpts,selected_new_candidate]) # C as per problem statement
                fir_obs_C = np.vstack([fir_obs_C, selected_new_candidate]) # F as per problem statement
                rvec_candidate = np.vstack([rvec_candidate,new_candidate_rvec]) # T as per problem statement (but only rotation)
                tvec_candidate = np.vstack([tvec_candidate, new_candidate_tvec]) # T as per problem statement (but only translation)
                
                
                temp_rvec= []
                temp_tvec = []
                temp_fir_obs_C = []
                temp_candidate_kpts = []
                
                for idx in range(candidate_kpts.shape[0]):
                    if(len(candidate_kpts.shape)!=2):
                        break
                    
                    a = candidate_kpts[idx,:]
                    min_norm1 = 100000000
                    min_norm2 = 100000000
                    
                    for j in range(candidate_kpts.shape[0]):
                        b = candidate_kpts[j,:]
                        norm = np.linalg.norm(a-b)
                        if (min_norm1 > norm) &(j!=idx):
                            min_norm1 = norm
                            
                    for j in range(good_img_keypoints.shape[0]):
                        b = good_img_keypoints[j,:]
                        norm = np.linalg.norm(a-b)
                        if min_norm2 > norm:
                            min_norm2= norm
                            
                    min_norm = max(min_norm1,min_norm2)
                    
                    if min_norm > 20:
                            temp_rvec.append(rvec_candidate[idx])
                            temp_tvec.append(tvec_candidate[idx])
                            temp_fir_obs_C.append(fir_obs_C[idx])
                            temp_candidate_kpts.append(candidate_kpts[idx])

                candidate_kpts = np.array(temp_candidate_kpts)
                rvec_candidate = np.array(temp_rvec)
                tvec_candidate = np.array(temp_tvec)
                fir_obs_C = np.array(temp_fir_obs_C)

                # get bearing angle b/w candidates
                angles = np.zeros([candidate_kpts.shape[0],1]).astype('float32')
                for l in range(candidate_kpts.shape[0]):
                    R_first, _ = cv2.Rodrigues((rvec_candidate[l,:]).T)
                    R_now,_ = cv2.Rodrigues(rvec) 
                             
                    a1_1 = np.vstack([(fir_obs_C[l,:][None]).T,1])
                    a2_2 = np.vstack([(candidate_kpts[l,:][None]).T,1])
                    
                    # normalize
                    a1_1 = np.linalg.inv(self.K) @ a1_1
                    a2_2 = np.linalg.inv(self.K) @ a2_2
                    
                    a2 = R_now @ a2_2
                    a1 = R_first @ a1_1
                    
                    temp = ((np.dot(a1.T,a2)))/(np.linalg.norm(a1)*np.linalg.norm(a2))                    
                    
                    if temp >=1:
                        temp = 1
                    elif temp<=-1:
                        temp = -1
                        
                    angles[l] = abs(math.acos(temp))
                    
                #if that angle is above a certain threshold, add it to the good_img_keypoints
                threshold = self.config["angle_threshold"]/180*np.pi
                
                if good_img_keypoints.shape[0] > 100:
                    threshold = threshold
                else:
                    threshold = 1/180*np.pi
                    
                
                index = np.where(angles >= threshold)
                
                for l in range(min(index[0].shape[0],200)):
                    
                    # This step is triangulation of new landmark from candidate
                    idx = index[0][l]
                    
                    R_first,_ = cv2.Rodrigues((rvec_candidate[idx,:]))
                    R_now,_ = cv2.Rodrigues(rvec)
 
                    t_first = (tvec_candidate[idx,:][None]).T                  
                                                   
                    t1 = -R_first.T @ t_first
                    t2 = -R_now.T @ tvec
                    
                    M1 = np.concatenate((R_first.T,t1), axis = 1)
                    M2 = np.concatenate((R_now.T,t2), axis = 1)   

                    inliers1 = np.array(fir_obs_C[idx,:])[None]
                    inliers2 = np.array(candidate_kpts[idx,:])[None]            
                  
                    inliers1 = np.vstack((inliers1.T, 1.0))
                    inliers2 = np.vstack((inliers2.T, 1.0))

                    norm_inliers1 = np.linalg.inv(self.K) @ inliers1
                    norm_inliers2 = np.linalg.inv(self.K) @ inliers2
                    
                    points3D = cv2.triangulatePoints(M1, M2, norm_inliers1[:2,:], norm_inliers2[:2,:])
                    points3D /= points3D[3]
                    
                    # t_cam is points3d in view of the camera frame (0,0,0 at camera)
                    t_cam = R_first.T @ points3D[0:3] - R_first.T @ t_first
                    if (t_cam[2] > 0 ):    
                        good_img_keypoints = np.vstack([good_img_keypoints,candidate_kpts[idx,:]])
                        good_img_landmarks = np.vstack([good_img_landmarks,(points3D[0:3]).T]) 

                candidate_kpts = np.delete(candidate_kpts,index[0],axis = 0)
                rvec_candidate = np.delete(rvec_candidate,index[0],axis = 0)
                tvec_candidate = np.delete(tvec_candidate,index[0],axis = 0)
                fir_obs_C = np.delete(fir_obs_C,index[0],axis =0)

                if self.config["remove_similar"]:
                    temp_good_img_keypoints = []
                    temp_good_img_landmarks = []
                    for idx in range(good_img_keypoints.shape[0]):
                        if(len(good_img_keypoints.shape)!=2):
                            break
                        
                        a = good_img_keypoints[idx,:]
                        min_norm = 100000000
                        
                        for j in range(good_img_keypoints.shape[0]):
                            b = good_img_keypoints[j,:]
                            norm = np.linalg.norm(a-b)
                            if (min_norm > norm) &(j!=idx):
                                min_norm = norm
                        
                        if min_norm > 3:
                            temp_good_img_keypoints.append(good_img_keypoints[idx])
                            temp_good_img_landmarks.append(good_img_landmarks[idx])

                    good_img_keypoints = np.array(temp_good_img_keypoints)
                    good_img_landmarks = np.array(temp_good_img_landmarks)

            p0 = good_img_keypoints.reshape(-1,1,2) # P as per problem statement
            candidate_kpts_obj = self.h.kpts2kpts2Object(candidate_kpts)
            output_image1 = cv2.drawKeypoints(cv2.cvtColor(self.images[i], cv2.COLOR_GRAY2BGR), candidate_kpts_obj, 0, (255,0,0))

            good_img_kpts_obj = self.h.kpts2kpts2Object(good_img_keypoints)
            output_image2 = cv2.drawKeypoints(output_image1, good_img_kpts_obj, 0, (0,255,0))

            ax[0,1].scatter(T_X[-1], T_Z[-1], c='#ff0000', s=3) #row=0, col=0
            ax[1,1].scatter(T_X[-1], T_Z[-1], c='#ff0000', s=3) #row=0, col=0

            points = ax[1,1].scatter(good_img_landmarks[:,0],good_img_landmarks[:,2],c='#000000', s=1) #row=1, col=1
            #ax[0,1].scatter(good_img_landmarks[:,0],good_img_landmarks[:,2],c='#000000', s=1) #row=1, col=1
            ax[1,0].bar(i, len(good_img_landmarks), color="#000000")
            ax[0,0].imshow(output_image2)
            xmin = min(T_X[-20:])-self.config["plot_x_scale"][0]
            xmax = max(T_X[-20:])+self.config["plot_x_scale"][1]
            ymin = min(T_Z[-20:])-self.config["plot_y_scale"][0]
            ymax = max(T_Z[-20:])+self.config["plot_y_scale"][1]
            if(i>21):
                ax[1,1].axis(xmin=xmin, xmax = xmax, ymin=ymin, ymax=ymax)
                ax[1,0].axis(xmin=i-20, xmax= i)

            ax[0,0].set_title("Image")
            ax[0,1].set_title("Full trajectory", fontsize=7)
            ax[1,0].set_title("Num of kpts detected", fontsize=7)
            ax[1,1].set_title("Trajectory of last 20 frames", fontsize=7)

            plt.pause(0.05)
            points.remove()
            candidate_kpts = candidate_kpts.reshape(-1,1,2)

        plt.show()
        # self.h.generate_trajectory(list(zip(T_X, T_Z)))