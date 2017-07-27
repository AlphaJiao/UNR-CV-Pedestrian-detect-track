#!/usr/bin/env python
import cv2
import rospy
import numpy as np
import scipy as sp
from scipy.linalg import norm
from sensor_msgs.msg import Image
from imutils.object_detection import non_max_suppression
import imutils 
from cv_bridge import CvBridge, CvBridgeError
from filterpy.monte_carlo import systematic_resample
from particle import ParticleFilter

N_PARTICLES = 10

class Tracker:
    def __init__(self):
        # Init publishers
        self.pub_ptcls = rospy.Publisher('tracker/ptcls',Image, queue_size=50)
    
        # Init BG Subtractors and cv_bridge
        self.fgbg      = cv2.createBackgroundSubtractorMOG2()
        self.fgbg_edge = cv2.createBackgroundSubtractorMOG2()
        self.bridge    = CvBridge()
    
        # Init opflow
        self.prev_cv_im = None
        self.initial_flag = None
        self.old_of = None
        self.roi = None
        #self.p0 = np.empty([10,1,2],dtype=np.float32)
        self.p0 = None
        self.feature_params = dict( maxCorners = 10,
                               qualityLevel = 0.0001,
                               minDistance = 5,
                               blockSize = 11 )
        self.lk_params = dict( winSize  = (17, 17),
                          maxLevel = 3,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.pfilter = ParticleFilter()
        self.particles = [None]
        self.old_particles = self.particles
        self.weights = np.zeros(N_PARTICLES)
        self.xs = []
        
    
    def imu_listener(self):
        # Init ROS node
        rospy.init_node('pf_tracker')
        self.sub = rospy.Subscriber("/usb_cam/image_raw", Image, self.track)
        rospy.spin()

    def track(self, msg):
        try:
            # Image preprocessing - ROS -> OPENCV bridging and downscaling
            cv_im = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
            #cv_im_down = self.downscale(cv_im) 
            cv_im_down = cv_im
            (rects, weights) = hog.detectMultiScale(cv_im_down, winStride=(4,4),padding=(32,32),scale=1.2)
            rects = np.array([[x, y, x + w, y + h]for (x, y, w, h) in rects])
            pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(cv_im_down, (xA, yA), (xB, yB), (0, 255, 0),1)
                xc = (xA+xB)/2
                yc = (yA+yB)/2
                self.roi = cv_im_down[yc-50:yc+50,xc-20:xc+20]
                of_raw2 = self.opflow(self.roi)
                of_im2  = self.of_image_conv(of_raw2)
                fg_im, bg_im, fg_mask, bg_mask = self.bg_subtract(self.roi, self.fgbg)
                ptcl_im = self.resample(self.roi, fg_mask)
            
            # Print particles
                for p in self.particles:
                    cv2.circle( ptcl_im, (p[0].astype('float32'), p[1].astype('float32')),2,(0,255,255),4)
                    #cv_im_down[y:y+8,x:x+4]=self.roi
                    cv_im_down[yc-50:yc+50,xc-20:xc+20]=ptcl_im
                    try:
                        self.publish(cv_im_down)
                    except CvBridgeError as e:
                        print e
        except CvBridgeError as e:
            print e

    def n_not_equal(self, A, B):
        flag = np.array([np.not_equal(a, b) for a, b in zip(A,B)]).ravel()
        return np.any( flag )
    
    def resample(self, img, thresh_im):
        # If particles are unset, initialize particle filter
        if np.all(self.particles[0] == None):
            print "Sampling Initial Particles"
            self.particles = self.pfilter.create_uniform_particles((0,img.shape[0]), (0,img.shape[1]), (0, 2*np.pi), N_PARTICLES)
            self.old_particles = self.particles
            (good_new, good_old) = self.getFeaturePoints(img, thresh_im) 
        else:
            (good_new, good_old) = self.getFeaturePoints(img, thresh_im) 

        # Get feature points
        if np.all(good_new != None) and len(good_new) > 1:
            good_new = good_new.reshape(-1,2)
            
            # Predict
            mean    = np.mean(good_new, axis=0) 
            std_dev = np.std( good_new, axis=0) 
            self.pfilter.predict(self.particles, u=mean, std=std_dev)

            # Update
            std_err = np.std(good_new, axis=0)
            zs = np.zeros((len(good_new), len(self.particles)))
            for f_ind, feature in enumerate(good_new):
                for p_ind, particle in enumerate(self.particles[:,0:2]):
                    zs[f_ind,p_ind] += np.linalg.norm(particle - feature)
                

            self.pfilter.update(self.particles,
                                    self.weights,
                                    z=zs, 
                                    R=np.array(std_err),
                                    landmarks=good_new)


            # Resample if too few effective particles
            if self.pfilter.neff(self.weights) < N_PARTICLES/2:
                indices = systematic_resample(self.weights)
                self.pfilter.resample(self.particles, self.weights, indices)


            # Estimate position
            mu, var = self.pfilter.estimate(self.particles, self.weights)
            self.xs.append(mu)

        
        ptcl_im = img.copy()
        if np.all(good_new != None) and len(good_new) > 1:
            for f in good_new:
                cv2.circle( ptcl_im, (f[0].astype('float32'), f[1].astype('float32')),2,(0,0,255),4)


        return ptcl_im

    def getFeaturePoints(self, frame, mask_im):

        # If initial points are unset, initialize optical flow
        if np.all(self.p0 == None):
            print "Sampling Initial Features"
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.old_frame = frame
            old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
            self.p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
        else:
            self.old_frame = frame
            # Set frames to use for feature points
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            old_gray = cv2.cvtColor(self.old_frame, cv2.COLOR_BGR2GRAY)
        # Calculate LK optical flow from Shi/Tomasi features
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, self.p0, None, **(self.lk_params))
         
        # If there aren't good features, make new ones
        if np.all(st==0) or np.all(p1 == None):
            tmp_features = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
            if np.all(tmp_features != None):
                self.p0 = tmp_features.reshape(-1,2)
            return (self.p0, self.p0)

        # Select good points
        p1 = p1.reshape(-1, 1, 2)
        good_new = np.array(p1[st==1])
        good_old = np.array(self.p0[st==1])
        
        # Now update the previous frame and previous points
        self.old_frame = frame.copy()
        self.p0 = good_new.reshape(-1,1,2)


        # If there aren't good features, make new ones
        if np.all(good_new == None) or len(good_new) == 0:
            tmp_features = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
            if np.all(tmp_features != None):
                self.p0 = tmp_features.reshape(-1,1,2)
            return (self.p0, self.p0)


        good_new = np.array(good_new)
        inbounds_arr = np.array([np.less(p, np.array(mask_im.shape)) for p in good_new])
        bound_mask =  np.logical_and(inbounds_arr[:,0], inbounds_arr[:,1])
        good_new = good_new[bound_mask]
        good_old = good_old[bound_mask]
        passed_mask = np.where([mask_im[int(p[0]), int(p[1])] > 0 for p in good_new])
        good_new = good_new[passed_mask]
        good_old = good_old[passed_mask]

        # If there aren't good features, make new ones
        if len(good_old) == 0:
            tmp_features = cv2.goodFeaturesToTrack(old_gray, mask = mask_im, **(self.feature_params))
            if np.all(tmp_features != None):
                self.p0 = tmp_features.reshape(-1,1,2)
            return (self.p0, self.p0)
        return (good_new.reshape(-1,1,2), good_old)


    def bg_subtract(self, im, bg_subtractor):
        fg_mask = bg_subtractor.apply(im)
        bg_mask = cv2.bitwise_not(fg_mask)
        fg_im   = cv2.bitwise_and(im,im,mask = fg_mask)
        bg_im   = cv2.bitwise_and(im,im,mask = bg_mask)
        return (fg_im, bg_im, fg_mask, bg_mask)


    #optical flow computation
    def opflow(self, next_image):
        next_cv_im = cv2.cvtColor(next_image,cv2.COLOR_BGR2GRAY)
        if np.all(self.prev_cv_im == None):
            self.prev_cv_im = next_cv_im
        if np.all(self.old_of == None):
            initial_flag = 0
        else:
            initial_flag = cv2.OPTFLOW_USE_INITIAL_FLOW
        of_val = cv2.calcOpticalFlowFarneback(self.prev_cv_im,next_cv_im, self.old_of,.5, 3,15, 2, 7, 1.1, initial_flag)  
        self.prev_cv_im = next_cv_im
        self.old_of = of_val
        return of_val

    #conversion from optical flow rawdata to image
    def of_image_conv(self, opflow_raw):
        h,w = opflow_raw.shape[:2]
        fx,fy = opflow_raw[:,:,0], opflow_raw[:,:,1]
        ang = np.arctan2(opflow_raw[:,:,0],opflow_raw[:,:,1])+np.pi
        v = np.sqrt(fx*fx + fy*fy)
        hsv = np.zeros((h,w,3),np.uint8)
        hsv[...,1] = 255
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,2] = cv2.normalize(v,None,0,255,cv2.NORM_MINMAX)
        return hsv

    def publish(self,cv_im_down): 
        self.pub_ptcls.publish(self.bridge.cv2_to_imgmsg(cv_im_down,"bgr8"))
        

if __name__ == '__main__':
    tracker = Tracker()
    tracker.imu_listener()

