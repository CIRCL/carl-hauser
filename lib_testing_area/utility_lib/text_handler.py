import logging
import pathlib
import time

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
from PIL import Image
import pytesseract

import configuration
from utility_lib import picture_class

MIN_CONFIDENCE = 0.5
REDUCE_FACTOR = 2


class Text_handler():
    def __init__(self, conf: configuration.Default_configuration):
        self.conf = conf
        self.logger =  logging.getLogger('__main__.' + __name__)

    def length_to_32_multiple(self, length, reduce_factor=REDUCE_FACTOR):
        # logging.debug(f"Input length : {length}")

        reduced = int(length / reduce_factor)
        logging.debug(f"Reduced length : {reduced}")

        round = 32 * ((reduced // 32) + 1)
        logging.debug(f"Rounded to 32 length : {round}")

        return round

    def get_hist_count_colors(self, img, startX, startY, endX, endY):
        crop_img = img[startX:endX, startY:endY]  # H,W

        a2D = crop_img.reshape(-1, crop_img.shape[-1])
        col_range = (256, 256, 256)  # generically : a2D.max(0)+1
        a1D = np.ravel_multi_index(a2D.T, col_range)
        return np.unravel_index(np.bincount(a1D).argmax(), col_range)

    def get_mean_color(self, img, startX, startY, endX, endY):
        percentage_more = 0.5

        startX_less = int(startX * (1 - percentage_more))
        endX_more = int(endX * (1 + percentage_more))
        startY_less = int(startY * (1 - percentage_more))
        endY_more = int(endY * (1 + percentage_more))

        crop_img = img[startX_less:endX_more, startY_less:endY_more]  # H,W

        # calculate the average color of each row of our image
        avg_color_per_row = np.average(crop_img, axis=0)

        # calculate the averages of our rows
        avg_colors = np.average(avg_color_per_row, axis=0)

        # avg_color is a tuple in BGR order of the average colors
        # but as float values
        # print(f'avg_colors: {avg_colors}')

        # so, convert that array to integers
        int_averages = np.array(avg_colors, dtype=np.uint8)
        # print(f'int_averages: {int_averages}')

        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)

        return int_averages

    def get_random_color(self, boxes, orig):
        color = []
        if len(boxes) == 0:
            return [0, 0, 0]
        try:
            curr_box = boxes[0]
            color1 = self.get_hist_count_colors(orig, curr_box[0], curr_box[1], curr_box[2], curr_box[3])
            curr_box = boxes[int(len(boxes) / 2)]
            color2 = self.get_hist_count_colors(orig, curr_box[0], curr_box[1], curr_box[2], curr_box[3])
            curr_box = boxes[len(boxes) - 1]
            color3 = self.get_hist_count_colors(orig, curr_box[0], curr_box[1], curr_box[2], curr_box[3])
            color = [np.mean([color1[0], color2[0], color3[0]]), np.mean([color1[1], color2[1], color3[1]]),
                     np.mean([color1[2], color2[2], color3[2]])]
        except Exception as e:
            logging.error(f"Error during random color getting {e}")
            return [0, 0, 0]

        return color

    def picture_to_cv2_picture(self, picture: picture_class.Picture):
        return cv2.imread(str(picture.path.resolve()))

    '''
    level	page_num	block_num	par_num	line_num	word_num	left	top	    width	height	conf	text
    1	    1	        0	        0	    0	        0	        0	    0	    1366	2240	-1	
    2	    1	        1	        0	    0	        0	        152	    18	    46	    13	    -1	
    3	    1	        1	        1	    0	        0	        152	    18	    46	    13	    -1	
    4	    1	        1	        1	    1	        0	        152	    18	    46	    13	    -1	
    5	    1	        1	        1	    1	        1	        152	    18	    46	    13	    96	    English
    2	    1	        2	        0	    0	        0	        603	    64	    160 	26	    -1	
    3	    1	        2	        1	    0	        0	        603	    64	    160 	26	    -1	
    4	    1	        2	        1	    1	        0	        603	    64	    160	    26	    -1	
    5	    1	        2	        1	    1	        1	        603	    64	    25	    25	    75	    G
    5	    1	        2	        1	    1	        2	        634	    65	    129	    25	    82	    ackdupexm
    '''

    @staticmethod
    def clean_boxes(boxes):
        THREESHOLD = 60
        tmp_boxes = []

        # Threeshold on confidence
        for i in boxes :
            conf = int(i[10])
            startX = int(i[6])
            startY = int(i[7])
            width = int(i[8])
            height = int(i[9])
            endX = startX + width
            endY = startY + height
            if conf != -1 and conf > THREESHOLD and height <= 30:
                tmp_boxes.append(["",startX,startY,endX,endY,0])

        return tmp_boxes

    def extract_text_Tesseract(self, picture: picture_class.Picture):
        # img = cv2.imread(r'/<path_to_image>/digits.png')
        # print("test : " + pytesseract.image_to_string(Image.open(picture.path)))
        # OR explicit beforehand converting
        # print(pytesseract.image_to_string(cv2.Image.fromarray(img)))
        image = self.picture_to_cv2_picture(picture)
        (H, W) = image.shape[:2]
        orig = image.copy()
        print("Look up boxes ! ")

        # boxes = pytesseract.image_to_boxes(Image.open(picture.path))
        boxes = pytesseract.image_to_data(Image.open(picture.path))
        if boxes != "" :
            boxes = [i.split("\t") for i in boxes.split("\n")]

        boxes = self.clean_boxes(boxes[1:len(boxes)-1])

        print(boxes)

        for (letter, startX, startY, endX, endY, z) in boxes:
            print("Box ! ")
            # startX, startY, endX, endY = int(startX), H-int(startY), int(endX), H-int(endY) # H- due to tesseract coordinate from bottom left corner
            startX, startY, endX, endY = int(startX), int(startY), int(endX), int(endY) # H- due to tesseract coordinate from bottom left corner
            # draw the bounding box on the image
            # orig = self.utility_rect_area(orig, startX, startY, endX, endY)
            # orig = self.fill_area_most_common_background(orig, startX, startY, endX, endY)
            # orig = self.fill_area_color(orig, startX, startY, endX, endY, color)
            orig = self.fill_area_black(orig, startX, startY, endX, endY)
            # orig = self.blur_area(orig, int(startX),int( startY),int( endX), int(endY))

        '''
        Cel esttr Moss: e eer M LP ent tce
        TNC CMON TICE
        E 152 2212 154 2222 0
        n 154 2212 157 2222 0
        g 161 2212 167 2219 0
        l 169 2209 175 2219 0
        '''

        return orig

    def extract_text_DeepModel(self, picture: picture_class.Picture):
        # load the input image and grab the image dimensions
        image = self.picture_to_cv2_picture(picture)
        orig = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        # SIZE DEPENDING ON ORIGINAL SIZE, MULTIPLE OF 32

        # (newW, newH) = (self.length_to_32_multiple(W),self.length_to_32_multiple(H))
        (newW, newH) = (640, 640)

        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]

        # load the pre-trained EAST text detector
        print("[INFO] loading EAST text detector...")
        net = cv2.dnn.readNet(str(pathlib.Path("./utility_lib/models/frozen_east_text_detection.pb").resolve()))

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        # show timing information on text prediction
        print("[INFO] text detection took {:.6f} seconds".format(end - start))
        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []
        # loop over the number of rows
        for y in range(0, numRows):
            # extract the scores (probabilities), followed by the geometrical
            # data used to derive potential bounding box coordinates that
            # surround text
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                # if our score does not have sufficient probability, ignore it
                if scoresData[x] < MIN_CONFIDENCE:
                    continue

                # compute the offset factor as our resulting feature maps will
                # be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then
                # compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volume to derive the width and height of
                # the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x, y)-coordinates for
                # the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probability score to
                # our respective lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        boxes = non_max_suppression(np.array(rects), probs=confidences)

        color = self.get_random_color(boxes, orig)

        # loop over the bounding boxes
        for (startX, startY, endX, endY) in boxes:
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)


            # draw the bounding box on the image
            # orig = self.utility_rect_area(orig, startX, startY, endX, endY)
            # orig = self.fill_area_most_common_background(orig, startX, startY, endX, endY)
            # orig = self.fill_area_color(orig, startX, startY, endX, endY, color)
            orig = self.blur_area(orig, startX, startY, endX, endY)

        return orig

    def utility_rect_area(self, img, startX, startY, endX, endY):
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return img

    def fill_area_white(self, img, startX, startY, endX, endY):
        cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 255), thickness=-1)
        return img

    def fill_area_black(self, img, startX, startY, endX, endY):
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 0), thickness=-1)
        return img

    def fill_area_color(self, img, startX, startY, endX, endY, color):
        cv2.rectangle(img, (startX, startY), (endX, endY), (color[0], color[1], color[2]), thickness=-1)
        return img

    def fill_area_mean_background(self, img, startX, startY, endX, endY):
        color = self.get_mean_color(img, startX, startY, endX, endY)
        cv2.rectangle(img, (startX, startY), (endX, endY), (int(color[0]), int(color[1]), int(color[2])), thickness=-1)
        return img

    def fill_area_most_common_background(self, img, startX, startY, endX, endY):
        color = self.get_hist_count_colors(img, startX, startY, endX, endY)
        cv2.rectangle(img, (startX, startY), (endX, endY), (int(color[0]), int(color[1]), int(color[2])), thickness=-1)
        return img

    def blur_area(self, img, startX, startY, endX, endY):
        # imS = cv2.resize(img, (540, 540))
        # cv2.imshow('img Blur', imS)

        tmp_pic = img[startY:endY, startX:endX]

        # imS = cv2.resize(tmp_pic, (540, 540))
        # cv2.imshow('tmp_pic Blur', imS)

        blur = cv2.GaussianBlur(tmp_pic, (25, 25), 0)

        # imS = cv2.resize(blur, (540, 540))
        # cv2.imshow('blur Blur', imS)

        img[startY:endY, startX:endX] = blur

        # imS = cv2.resize(img, (540, 540))
        # cv2.imshow('img Blur 2', imS)

        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()

        return img
