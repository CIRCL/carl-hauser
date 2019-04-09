import logging
import pathlib
import time

import cv2
import numpy as np
from imutils.object_detection import non_max_suppression

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

    def extract_text(self, picture: picture_class.Picture):
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
            orig = self.fill_area_color(orig, startX, startY, endX, endY, color)
            # orig = self.blur_area(orig, startX, startY, endX, endY)

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
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        return img
