import cv2
import numpy as np
import sys
ptB = []
ptD = []
ptF = []
ptE = []
max_ptB = []
max_ptD = []
max_ptF = []
max_ptE = []

class Stitcher1:
    
    def __init__(self):
        image1 = cv2.imread('set38/1.JPG')
        image2 = cv2.imread('set38/2.JPG')
        image3 = cv2.imread('set38/3.JPG')
        image4 = cv2.imread('set38/4.JPG')
#        image5 = cv2.imread('result/stitch1.jpg')
#        image6 = cv2.imread('result/stitch2.jpg')
#        image6 = cv2.resize(image6, (1570,900))
        image1 = cv2.resize(image1, (1000,750))
        image2 = cv2.resize(image2, (1000,750))
        image3 = cv2.resize(image3, (1000,750))
        image4 = cv2.resize(image4, (1000,750))
#        image5 = cv2.resize(image5, (1000,1500))
#        image6 = cv2.resize(image6, (1000,1500))
        
        
        self.stitch(image1, image2)
        self.stitch2(image3, image4)
#        self.stitch3()
#        self.findKeyPoints(image1)
        
    def stitch(self, image1, image2):

        (kpsA, featuresA) = self.findKeyPoints(image1)
        (kpsB, featuresB) = self.findKeyPoints(image2)
        M = self.matchedKeyPoints(kpsA, kpsB, featuresA, featuresB)

        if M is None:
            return None
        
        (matches, status, H) = M

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptB.append(kpsB[trainIdx][1])
        
        max_ptB = sorted(ptB, reverse=True)
#        print(max_ptB)
        sum_ptB = 0
        for i in range(9): 
            sum_ptB = sum_ptB + max_ptB[i]
        avg_ptB = sum_ptB/10
        avg_ptB = int(avg_ptB)

        cropImg = image2[:,avg_ptB:]

        connect = np.hstack((image1, cropImg))
        cv2.imwrite("result_set38/stitch1.jpg", connect)
#        self.stitch3()
        

 
    def stitch2(self, image3, image4):
        (kpsC, featuresC) = self.findKeyPoints(image3)
        (kpsD, featuresD) = self.findKeyPoints(image4)
        M = self.matchedKeyPoints(kpsC, kpsD, featuresC, featuresD)
        (matches, status, H) = M
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptD.append(kpsD[trainIdx][1])
        
        max_ptD = sorted(ptD, reverse=True)
        sum_ptD = 0
        for i in range(9): 
            sum_ptD = sum_ptD + max_ptD[i]
        avg_ptD = sum_ptD/10
        avg_ptD = int(avg_ptD)

        cropImg2 = image4[:,avg_ptD:]
        connect2 = np.hstack((image3, cropImg2))
        cv2.imwrite("result_set38/stitch2.jpg", connect2)

        self.stitch3()
        


    def stitch3(self):
        image5 = cv2.imread('result_set38/stitch1.jpg')
        image6 = cv2.imread('result_set38/stitch2.jpg')
        image5 = cv2.resize(image5, (2000,1500))
        image6 = cv2.resize(image6, (2000,1500))
        (kpsE, featuresE) = self.findKeyPoints(image5)
        (kpsF, featuresF) = self.findKeyPoints(image6)
        M = self.matchedKeyPoints(kpsE, kpsF, featuresE, featuresF)
        (matches, status, H) = M
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                ptF.append(kpsF[trainIdx][0])
        
        max_ptF = sorted(ptF, reverse=True)
        print(max_ptF)
        sum_ptF = 0
        for i in range(9): 
            sum_ptF = sum_ptF + max_ptF[i]
        avg_ptF = sum_ptF/10
        avg_ptF = int(avg_ptF)

        cropImg3 = image6[:,avg_ptF:]
        connect3 = np.hstack((image5, cropImg3))
 #       cv2.imshow("cropImg3", cropImg3)
 #       cv2.waitKey(0)

        self.resultimage(connect3)
          
    def findKeyPoints(self, image):

        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)

        kps = np.float32([kp.pt for kp in kps])
#        print("kps"+ str(kps))
#        print("feature"+str(features.shape))
        return (kps, features)
    
    def matchedKeyPoints(self, kpsA, kpsB, featuresA, featuresB):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_,i) in matches])
            ptsB = np.float32([kpsB[i] for (i,_) in matches])
            #print(ptsB)
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
               
            return (matches, status, H)
        return None

    def resultimage(self, result):
 
       cv2.imshow("Result", result)
       cv2.imwrite("result_set38/finalstitch.jpg", result)
       cv2.waitKey(0)
            
    def test(self):
        print("TEST!")
        
 
test = Stitcher1()


