import numpy as np
import cv2
import math
from imutils.object_detection import non_max_suppression
import pytesseract
import time
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

import time


cap = cv2.VideoCapture(0)



while(cap.isOpened()):
    ret, frame = cap.read()

    #resize is happening here. Video resize not working
    #frame = cv2.resize(frame, (framevideoW, framevideoH))

    if ret==True:
        #frame = cv2.flip(frame,0)

        # resize the frame, maintaining the aspect ratio
        #frame = imutils.resize(frame, width=1000)
        orig = frame.copy()



        # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
        # You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan`
        # to switch the language model in order.
        ocr = PaddleOCR(use_angle_cls=True, lang='en',show_log = False)  # need to run only once to download and load model into memory
        #img_path = 'PaddleOCR/doc/imgs_en/img_12.jpg'

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = ocr.ocr(frame, cls=True)

        if len(result) != 0:
            for line in result:
                print(line)
            print("---------------------------- PaddleOCR --------------------------------------------------------------")
            print("\n")
            print("\n")

        #image = Image.open(frame).convert('RGB')
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(frame, boxes, txts, scores, font_path='simfang.ttf')
        im_show = Image.fromarray(im_show)
        im_show.save('result.jpg')

        line_length = 15
        # frame = draw_border(frame, boxes[0], boxes[1], boxes[2], boxes[3], line_length)
        if len(boxes) != 0:
            try:
                for (box, score) in zip(boxes, scores):
                    box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
                    frame = cv2.polylines(np.array(frame), [box], True, (255, 0, 0), 2)
                #cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), -1)
                #cv2.drawContours(frame, np.array([boxes]), 0, (0, 0, 255), 2)
            except Exception as e:

                print(str(e))
        #rect = cv2.minAreaRect(boxes)

        # convert rect to 4 points format

        #box = cv2.boxPoints(rect)
        #box = np.int0(box)
        #cv2.drawContours(frame, np.array([box]), 0, (0, 0, 255), 2)


        #cv2.polylines(frame ,np.array([boxes]), True,(0,255,255 ))

        # show the output frame
        cv2.imshow("Text Detection", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()

cv2.destroyAllWindows()