import cv2
import time
import numpy as np

# 用 gpu來運行 keypoint 提取
device = "gpu" # please change it to "gpu" if the model needs to be run on cuda.

# 加入openpose 使用的節點和抓取節點的模組
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
nPoints = 18
# COCO Output Format
keypointsMapping = ['Nose', 'Neck', 'R-Sho', 'R-Elb', 'R-Wr', 'L-Sho', 
                    'L-Elb', 'L-Wr', 'R-Hip', 'R-Knee', 'R-Ank', 'L-Hip', 
                    'L-Knee', 'L-Ank', 'R-Eye', 'L-Eye', 'R-Ear', 'L-Ear']
#點

POSE_PAIRS = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7],
              [1,8], [8,9], [9,10], [1,11], [11,12], [12,13],
              [1,0], [0,14], [14,16], [0,15], [15,17],
              [2,17], [5,16] ]
#線

# index of pafs correspoding to the POSE_PAIRS
# e.g for POSE_PAIR(1,2), the PAFs are located at indices (31,32) of output, Similarly, (1,5) -> (39,40) and so on.

mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], 
          [19,20], [21,22], [23,24], [25,26], [27,28], [29,30], 
          [47,48], [49,50], [53,54], [51,52], [55,56], 
          [37,38], [45,46]]
#第二個人


colors = [ [0,100,255], [0,100,255], [0,255,255], [0,100,255], [0,255,255], [0,100,255],
         [0,255,0], [255,200,100], [255,0,255], [0,255,0], [255,200,100], [255,0,255],
         [0,0,255], [255,0,0], [200,200,0], [255,0,0], [200,200,0], [0,0,0]]
#grb

# 函數:取得照片節點
# Find the Keypoints using Non Maximum Suppression on the Confidence Map
#Non Maximum Suppression https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E7%89%A9%E4%BB%B6%E5%81%B5%E6%B8%AC-non-maximum-suppression-nms-aa70c45adffa
def getKeypoints(probMap,threshold=0.1):
    
    mapSmooth = cv2.GaussianBlur(probMap,(3,3),0,0)

    mapMask = np.uint8(mapSmooth>threshold)
    keypoints = []
    
    #find the blobs
    contours, _ = cv2.findContours(mapMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #for each blob find the maxima
    for cnt in contours:
        blobMask = np.zeros(mapMask.shape)
        blobMask = cv2.fillConvexPoly(blobMask, cnt, 1)
        maskedProbMap = mapSmooth * blobMask
        _, maxVal, _, maxLoc = cv2.minMaxLoc(maskedProbMap)
        keypoints.append(maxLoc + (probMap[maxLoc[1], maxLoc[0]],))

    return keypoints

# 函數:輸入照片轉節點後組成一組能進入lstm的list
def node_print(a):
    image1 = a
    frameWidth = image1.shape[1]
    frameHeight = image1.shape[0]
    #======================================================================================
    t = time.time()
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
        #print("Using CPU device")
    elif device == "gpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        #print("Using GPU device")

    # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    inWidth = int((inHeight/frameHeight)*frameWidth)

    inpBlob = cv2.dnn.blobFromImage(image1, 1.0 / 255, (inWidth, inHeight),
                              (0,0,0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()
    print("Time Taken = {}".format(time.time() - t))
    #==========================================================================================
    i = 0
    probMap = output[0, i, :, :]
    probMap = cv2.resize(probMap, (frameWidth, frameHeight))
    #plt.figure(figsize=[14,10])
    #plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    #plt.imshow(probMap, alpha=0.6)
    #plt.colorbar()
    #plt.axis("off")
    #============================================================================================
    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1
    table=[]
    text=[]
    for part in range(nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
    #     plt.figure()
    #     plt.imshow(255*np.uint8(probMap>threshold))
        keypoints = getKeypoints(probMap, threshold)
        #print("Keypoints - {} : {}".format(keypointsMapping[part], keypoints))


        table.append(keypoints)



        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)


        #print("=================================")
        if(len(keypoints)==0):
            text.append(0)
            text.append(0)
            text.append(0)
            text.append(0)
        elif(len(keypoints)==1):
            pp1=keypoints[0]
            text.append(pp1[0])
            text.append(pp1[1])
            text.append(0)
            text.append(0)
        elif(len(keypoints)>=2):
            pp1=keypoints[0]
            pp2=keypoints[1]
            text.append(pp1[0])
            text.append(pp1[1])
            text.append(pp2[0])
            text.append(pp2[1])
    
    #print(text)
    #print(len(text))
    return(text)

# 函數:相片捕捉，1秒6張，拍攝十秒，共60張
def img_capture():
    cap = cv2.VideoCapture(0)
    c=[]
    while len(c) != 60:
        ret,img=cap.read()
        if ret:
            
            cv2.imshow('img',img)
            c.append(img)
        if cv2.waitKey(166)==ord('q'):
        
            break 
        
    cap.release()
    cv2.destroyAllWindows()
    
    return (c)



