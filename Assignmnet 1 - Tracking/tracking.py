import glob
import math

import cv2
import torch
from matplotlib import pyplot as plt
from torchvision import ops
import os
import numpy as np
from numpy import array

## Declaring Variables
bounding_box = []  # values will be added by tracker algorithm as it tracks the object
truthBOX = []  # actual(TRUTH) bounding box that came with the video
mainFrameShape = []
iousMIL = []
iousKCF = []
iousMF = []
iousCSRT = []
timeMIL = []
timeKCF = []
timeMF = []
timeCSRT = []

tracker_types = ['MIL', 'KCF', 'MEDIANFLOW','CSRT']
# tracker_types = [  'CSRT' ]

for tracker_type in tracker_types:
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    bounding_box = []



    # video = cv2.VideoCapture("IMG_3001.mp4")
    # video = cv2.VideoCapture("chaplin.mp4")
    # video = cv2.VideoCapture("Tracking_boomerang.mp4")
    # video = cv2.VideoCapture(0)

    ok, frame = video.read()
    ################################# Calucuate TruthBOX from given points  #################################
    i=0
    mainFrameShape = frame.shape
    path = "/Users/umangkoul/Downloads/2nd Sem/CS 510/cs510tutorials/Assignmnet 1 - Tracking/" + "jerry"
    os.chdir(path)
    if not truthBOX:
        for filename in sorted((glob.glob('*.txt'))):
            # print(filename)
            with open(
                    '/Users/umangkoul/Downloads/2nd Sem/CS 510/cs510tutorials/Assignmnet 1 - Tracking/' + "jerry" + '/' + filename) as f:
                contents = f.read().splitlines()
            i+=1
            print(i)
            coor = contents[0].split(' ')[1:5]
            im_width = mainFrameShape[1]
            im_height = mainFrameShape[0]
            start_point_x = round((float(coor[0]) - 0.5 * float(coor[2])) * im_width)
            start_point_y = round((float(coor[1]) - 0.5 * float(coor[3])) * im_height)

            end_point_x = start_point_x + round(float(coor[2]) * im_width)
            end_point_y = start_point_y + round(float(coor[3]) * im_height)
            truthBOX.append([start_point_x, start_point_y, end_point_x, end_point_y])

    ################################# Calucuated TruthBOX  #################################
    # bbox = cv2.selectROI(frame, False)   ----- this was initial approach when I was manualy
    # selecting a box around the image but i realized that my box could be of wrong dimention and alter the IOU
    bbox=[truthBOX[0][0],truthBOX[0][1], truthBOX[0][2]-truthBOX[0][0],truthBOX[0][3]-truthBOX[0][1]]
    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)
    while True:
        #Time Noted
        startTime = cv2.getTickCount()
        ok, frame = video.read()
        if not ok:
            break
        ok, bbox = tracker.update(frame)
        # Tracking Done Note time again
        endTime=cv2.getTickCount()
        time_taken = (endTime - startTime)/ cv2.getTickFrequency()
        if tracker_type == 'MIL':
            timeMIL.append(time_taken)
        if tracker_type == 'KCF':
            timeKCF.append(time_taken)
        if tracker_type == 'MEDIANFLOW':
            timeMF.append(time_taken)
        if tracker_type == "CSRT":
            timeCSRT.append(time_taken)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            bounding_box.append([int(bbox[0]), int(bbox[1]), int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])])
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            # cv2.waitKey(300)
            cv2.putText(frame, "Object detected ! ! ! ", (100, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75,
                        (50, 170, 50), 2);

        else:
            # Lost tracking display on screen
            bounding_box.append([0, 0, 0, 0])
            cv2.putText(frame, "Tracking failure detected", (100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(frame, tracker_type + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
        cv2.imshow("Tracking", frame)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break

    mini= min(len(truthBOX),len(bounding_box))
    for idx  in range(mini):
        iou = ops.box_iou((torch.tensor([[truthBOX[idx][0], truthBOX[idx][1], truthBOX[idx][2], truthBOX[idx][3]]],
                                        dtype=torch.float)),
                          (torch.tensor([[bounding_box[idx][0], bounding_box[idx][1], bounding_box[idx][2],
                                          bounding_box[idx][3]]], dtype=torch.float)))
        if tracker_type == 'MIL':
            iousMIL.append(iou.numpy()[0][0])
        if tracker_type == 'KCF':
            iousKCF.append(iou.numpy()[0][0])
        if tracker_type == 'MEDIANFLOW':
            iousMF.append(iou.numpy()[0][0])
        if tracker_type == "CSRT":
            iousCSRT.append(iou.numpy()[0][0])
    # Exit main loop also if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27: break
    video.release()
    cv2.destroyAllWindows()
############################################ Tracking Program Ends ############################################
# print('[INFO] Max time Value in MIL         : ', np.max(timeMIL))
# print('[INFO] Max time Value in KCF         : ', np.max(timeKCF))
# print('[INFO] Max time Value in MEDIAN FLOW : ', np.max(timeMF))
# print('[INFO] Max time Value in CSRT        : ', np.max(timeCSRT))
# print("[INFO] Length of truthBOX            : ",len(truthBOX))
# print("[INFO] Length of bounding_box        : ",len(bounding_box))
allMax=[np.max(timeMIL),np.max(timeKCF),np.max(timeMF),np.max(timeCSRT)]
############################################ IOU Graph Program Starts ############################################
ax = plt.gca()
ax.set_ylim([0, 2])
# ax.set_prop_cycle(['red', 'green', 'blue', 'yellow'])
x_axis = array(range(len(truthBOX) + 0))
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("IoU Value", fontsize=12)
plt.text(10, 1.90, 'Mean IoU - MIL: '+str(np.mean(iousMIL)), fontsize=11)
plt.text(10, 1.80, 'Mean IoU - KCF: '+str(np.mean(iousKCF)), fontsize=11)
plt.text(10, 1.70, 'Mean IoU - Med.Flow: '+str(np.mean(iousMF)), fontsize=11)
plt.text(10, 1.60, 'Mean IoU - CSRT: '+str(np.mean(iousCSRT)), fontsize=11)

plt.text(10, 1.40, 'Std Dev IoU - MIL: '+str(np.std(iousMIL)), fontsize=11)
plt.text(10, 1.30, 'Std Dev IoU - KCF: '+str(np.std(iousKCF)), fontsize=11)
plt.text(10, 1.20, 'Std Dev IoU - Med.Flow: '+str(np.std(iousMF)), fontsize=11)
plt.text(10, 1.10, 'Std Dev IoU - CSRT: '+str(np.std(iousCSRT)), fontsize=11)
print("[INFO] Mean IoU - MIL         :  ",np.mean(iousMIL))
print("[INFO] Mean IoU - KCF         :  ",np.mean(iousKCF))
print("[INFO] Mean IoU - MedFLow     :  ",np.mean(iousMF))
print("[INFO] Mean IoU - CSRT        :  ",np.mean(iousCSRT))
print("[INFO] Std Dev IoU - MIL      :  ",np.std(iousMIL))
print("[INFO] Std Dev IoU - KCF      :  ",np.std(iousKCF))
print("[INFO] Std Dev IoU - MedFLow  :  ",np.std(iousMF))
print("[INFO] Std Dev IoU - CSRT     :  ",np.std(iousCSRT))
plt.title('Metric : Intersection over Union - IoU')
plt.plot(x_axis, iousMIL,label='MIL',color="aquamarine",linewidth=0.5)
plt.plot(x_axis, iousKCF,label='KCF',color="darkviolet",linewidth=0.5)
plt.plot(x_axis, iousMF,label='Med.Flow',color="cornflowerblue",linewidth=0.5)
plt.plot(x_axis, iousCSRT,label='CSRT',color="coral",linewidth=0.5)
plt.legend( loc='upper right')
plt.show()

############################################ Time Taken Graph Program Starts ############################################
ax = plt.gca()
ax.set_ylim([0, np.max(allMax)+0.35])
# ax.set_prop_cycle(['red', 'green', 'blue', 'yellow'])
x_axis = array(range(len(timeMIL) + 0))
plt.xlabel("Frame Number", fontsize=12)
plt.ylabel("Time taken to track particular frame ( seconds )", fontsize=12)
plt.text(.1, .99, 'Mean Time Taken - MIL: '+str(np.round(np.mean(timeMIL),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)
plt.text(.1, .95, 'Mean Time Taken - KCF: '+str(np.round(np.mean(timeKCF),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)
plt.text(.1, .91, 'Mean Time Taken - Med.Flow: '+str(np.round(np.mean(timeMF),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)
plt.text(.1, .87, 'Mean Time Taken - CSRT: '+str(np.round(np.mean(timeCSRT),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)

print("[INFO] Mean TIME - MIL        :  ",np.mean(timeMIL))
print("[INFO] Mean TIME - KCF        :  ",np.mean(timeKCF))
print("[INFO] Mean TIME - MedFLow    :  ",np.mean(timeMF))
print("[INFO] Mean TIME - CSRT       :  ",np.mean(timeCSRT))
print("[INFO] Std Dev TIME - MIL     :  ",np.std(timeMIL))
print("[INFO] Std Dev TIME - KCF     :  ",np.std(timeKCF))
print("[INFO] Std Dev TIME - MedFLow :  ",np.std(timeMF))
print("[INFO] Std Dev TIME - CSRT    :  ",np.std(timeCSRT))

plt.text(.1, .79, 'Std Dev Time Taken - MIL: '+str(np.round(np.std(timeMIL),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)
plt.text(.1, .75, 'Std Dev Time Taken - KCF: '+str(np.round(np.std(timeKCF),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)
plt.text(.1, .71, 'Std Dev Time Taken - Med.Flow: '+str(np.round(np.std(timeMF),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)
plt.text(.1, .67, 'Std Dev Time Taken - CSRT: '+str(np.round(np.std(timeCSRT),decimals=4)), fontsize=11,ha='left', va='top', transform=ax.transAxes)
plt.title('Metric : Time taken to track object in frame')
plt.plot(x_axis, timeMIL,label='MIL',color="aquamarine",linewidth=0.5)
plt.plot(x_axis, timeKCF,label='KCF',color="darkviolet",linewidth=0.5)
plt.plot(x_axis, timeMF,label='Med.Flow',color="cornflowerblue",linewidth=0.5)
plt.plot(x_axis, timeCSRT,label='CSRT',color="coral",linewidth=0.5)
# plt.legend(loc='best')
plt.legend( loc='upper right')
plt.show()

############################################ Time Taken Graph Program Ends ############################################
