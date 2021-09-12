import cv2
import numpy as np
import mediapipe as mp

class HandDetector:
    def __init__(self, mode=False, maxHands=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.minDetectionCon, self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.handLm = self.mpHands.HandLandmark
        self.tipId = [self.handLm.THUMB_TIP.value, self.handLm.INDEX_FINGER_TIP.value, self.handLm.MIDDLE_FINGER_TIP.value, self.handLm.RING_FINGER_TIP.value, self.handLm.PINKY_TIP.value]
    
    def findHands(self, img, position=True, landmarks=True, bBox=True):
        xList = []
        yList = []
        bBoxList = []
        self.lmList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                tempLmList = []
                tempBBoxList = []

                if landmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                else:
                    img.flags.writeable = False
                if position:
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        xList.append(cx)
                        yList.append(cy)
                        tempLmList.append([id, cx, cy])
                if bBox:
                    minX, minY, maxX, maxY = min(xList), min(yList), max(xList), max(yList)
                    tempBBoxList = minX, minY, maxX, maxY
                    cv2.rectangle(img, (minX-20, minY-20), (maxX+20, maxY+20), (0, 255, 0), 2)

                self.lmList.append(tempLmList)
                bBoxList.append(tempBBoxList)

        return self.lmList, bBoxList

    def findHands(self, img):
        self.lmList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            # get [x, y, z] and [left, right] hand label
            handLms = self.results.multi_hand_landmarks
            handedness = self.results.multi_handedness

            for i in range(len(handLms)):
                tempLmList = []

                # draw
                self.mpDraw.draw_landmarks(img, handLms[i], self.mpHands.HAND_CONNECTIONS)

                for id, lm in enumerate(handLms[i].landmark):
                    tempLmList.append([id, lm.x, lm.y, lm.z])

                self.lmList.append([handedness[i].classification[0].label, tempLmList])

        # return [left, right] hand, and [x, y, z] of 21 points
        return self.lmList

    def findPositions(self, img):
        self.lmList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            # get [x, y, z]
            handLms = self.results.multi_hand_landmarks

            for i in range(len(handLms)):
                tempLmList = []

                # draw
                self.mpDraw.draw_landmarks(img, handLms[i], self.mpHands.HAND_CONNECTIONS)

                for id, lm in enumerate(handLms[i].landmark):
                    tempLmList.append([id, lm.x, lm.y, lm.z])

                self.lmList.append(tempLmList)

        return self.lmList
    
    def findHandBox(self, img):
        self.lmList = []
        xList = []
        yList = []
        bBoxList = []

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            # get [x, y, z]
            handLms = self.results.multi_hand_landmarks

            for i in range(len(handLms)):
                tempLmList = []
                tempBBoxList = []

                # draw
                self.mpDraw.draw_landmarks(img, handLms[i], self.mpHands.HAND_CONNECTIONS)

                for id, lm in enumerate(handLms[i].landmark):
                    tempLmList.append([id, lm.x, lm.y, lm.z])
                    # bBox
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                
                minX, minY, maxX, maxY = min(xList)-20, min(yList)-20, max(xList)+20, max(yList)+20
                tempBBoxList = minX, minY, maxX, maxY
                
                npBox = np.array([minX, minY, maxX, maxY])
                if np.any(npBox <= 0) | np.any(npBox >= h) | np.any(npBox >= w):
                    return self.lmList, []

                cv2.rectangle(img, (minX, minY), (maxX, maxY), (0, 255, 0), 2)

                self.lmList.append(tempLmList)
                bBoxList.append(tempBBoxList)

        return self.lmList, bBoxList

    def fingersUp(self, handNo=0):
        fingerList = []

        if len(self.lmList) == 0:
            return fingerList

        srcList = self.lmList[handNo]
        
        # thumb
        if srcList[self.handLm.THUMB_MCP.value][1] < srcList[self.handLm.INDEX_FINGER_MCP.value][1]:
            # left hand
            isUp = srcList[self.tipId[0]][1] < srcList[self.tipId[0]-1][1]
        else:
            # right hand
            isUp = srcList[self.tipId[0]][1] > srcList[self.tipId[0]-1][1]
        fingerList.append(int(isUp))

        # other fingers
        for id in range(1, len(self.tipId)):
            isUp = srcList[self.tipId[id]][2] < srcList[self.tipId[id]-2][2]
            fingerList.append(int(isUp))
        
        return fingerList