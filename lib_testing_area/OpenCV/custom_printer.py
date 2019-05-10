from typing import List
import configuration
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import pathlib
import traceback

from utility_lib import filesystem_lib, printing_lib, picture_class, execution_handler, json_class

class Local_Picture(picture_class.Picture):

    def load_image(self, path):
        if path is None or path == "":
            raise Exception("Path specified void")
            return None
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from cv's BRG default color order to RGB

        return image

class Custom_printer(printing_lib.Printer):

    def save_pictures(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name : pathlib.Path):

        if configuration.PICTURE_SAVE_MODE.TOP3 in self.conf.SAVE_PICTURE_INSTRUCTION_LIST:
            try :
                new_directory = file_name.parent / "TOP3"
                if not new_directory.exists() :new_directory.mkdir()
                self.save_pictures_top_3(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_top3").with_suffix(".png"))
            except Exception as e :
                self.logger.error(f"Error during the saving of TOP3 for {target_picture.path.name}")
        if configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3 in self.conf.SAVE_PICTURE_INSTRUCTION_LIST:
            try:
                new_directory = file_name.parent / "MATCHES"
                if not new_directory.exists() :new_directory.mkdir()
                self.save_pictures_matches_top_3(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_matches").with_suffix(".png"))
            except Exception as e :
                self.logger.error(f"Error during the saving of TOP3 MATCHES for {target_picture.path.name}")
        if configuration.PICTURE_SAVE_MODE.FEATURE_MATCHES_TOP3 in self.conf.SAVE_PICTURE_INSTRUCTION_LIST:
            try:
                new_directory = file_name.parent / "MATCHES_INOUTLINERS"
                if not new_directory.exists() :new_directory.mkdir()
                self.save_pictures_matches_top_3_red_green(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_inoutliners").with_suffix(".png"))
            except Exception as e :
                self.logger.error(f"Error during the saving of TOP3 RED/GREEN for {target_picture.path.name}")
        if configuration.PICTURE_SAVE_MODE.RANSAC_MATRIX in self.conf.SAVE_PICTURE_INSTRUCTION_LIST :
            if self.conf.FILTER == configuration.FILTER_TYPE.RANSAC :
                try:
                    new_directory = file_name.parent / "RANSAC"
                    if not new_directory.exists() :new_directory.mkdir()
                    self.save_pictures_ransac(sorted_picture_list, target_picture, file_name=new_directory / pathlib.Path(file_name.with_suffix("").name + "_ransac").with_suffix(".png"))
                except Exception as e:
                    self.logger.error(f"Error during the saving of RANSAC for {target_picture.path.name} : {e}")
                    self.logger.error(traceback.print_tb(e.__traceback__))
            else :
                self.logger.warning("RANSAC filter not selected for computation, but launch configuration is asking to save RANSAC pictures. Aborting this instruction.")

    def save_pictures_matches_top_3(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name):

        max_width = 0
        total_height = 0
        NB_BEST_PICTURES = 3

        # Preprocess to remove target picture from matches
        offset = json_class.remove_target_picture_from_matches(sorted_picture_list, target_picture)

        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            max_width = max(target_picture.image.shape[1] + sorted_picture_list[i + offset].image.shape[1],max_width)
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i + offset].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            # Get the matches
            output = self.draw_matches(sorted_picture_list[i + offset], target_picture, sorted_picture_list[i + offset].matches)

            # img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
            # SAVE : tmp_img = Image.fromarray(outImg_good_matches)
            tmp_img = Image.fromarray(output)

            # Copy paste the matches in the column
            new_im.paste(tmp_img, (0, y_offset))

            # Add title
            self.add_text_matches(i + offset, sorted_picture_list[i + offset], target_picture, y_offset, max_width, draw)

            # Set next offset
            y_offset += tmp_img.size[1]

        self.logger.debug(f"Save to : {str(file_name)}")
        print(file_name)

        new_im.save(file_name)

    def save_pictures_matches_top_3_red_green(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name):

        max_width = 0
        total_height = 0
        NB_BEST_PICTURES = 3

        # Preprocess to remove target picture from matches
        offset = json_class.remove_target_picture_from_matches(sorted_picture_list, target_picture)

        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            max_width = max(target_picture.image.shape[1] + sorted_picture_list[i + offset].image.shape[1],max_width)
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i + offset].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            # Get the matches
            output = cv2.drawMatches(sorted_picture_list[i + offset].image, sorted_picture_list[i + offset].key_points,
                                     target_picture.image, target_picture.key_points,
                                     sorted_picture_list[i + offset].not_filtered_matches, None, matchColor = (255,0,0))  # Draw circles.
            output = cv2.drawMatches(sorted_picture_list[i + offset].image, sorted_picture_list[i + offset].key_points,
                                     target_picture.image, target_picture.key_points,
                                     sorted_picture_list[i + offset].matches, output,
                                     # sorted_picture_list[i + offset].not_filtered_matches, output,
                                     matchColor = (0,255,0), flags=1)  # Draw circles.
                                     # matchesMask = sorted_picture_list[i + offset].matchesMask, matchColor = (0,255,0), flags=1)  # Draw circles.

            tmp_img = Image.fromarray(output)

            # Copy paste the matches in the column
            new_im.paste(tmp_img, (0, y_offset))

            # Add title
            self.add_text_matches(i + offset, sorted_picture_list[i + offset], target_picture, y_offset, max_width, draw)

            # Set next offset
            y_offset += tmp_img.size[1]

        self.logger.debug(f"Save to : {str(file_name)}")
        print(file_name)

        new_im.save(file_name)

    @staticmethod
    def draw_matches(pic1: Local_Picture, pic2: Local_Picture, matches):
        return cv2.drawMatches(pic1.image, pic1.key_points, pic2.image, pic2.key_points, matches, None)  # Draw circles.

    @staticmethod
    def save_matches(pic1: Local_Picture, pic2: Local_Picture, matches, distance):
        outImg = Custom_printer.draw_matches(pic1, pic2, matches)
        # outImg = printing_lib.print_title(outImg, pic1.path.name + " " + pic2.path.name)
        logging.debug("./RESULTS/" + pic1.path.name)
        t = pic1.path.name + " TO " + pic1.path.name + " IS " + str(distance)
        plt.text(0, 0, t, ha='center', wrap=True)
        plt.imsave("./RESULTS/" + pic1.path.name, outImg)

    @staticmethod
    def print_matches(pic1: Local_Picture, pic2: Local_Picture, matches):
        outImg = Custom_printer.draw_matches(pic1, pic2, matches)
        plt.figure(figsize=(16, 16))
        plt.title('ORB Matching Points')
        plt.imshow(outImg)
        plt.show()
        # input()

    @staticmethod
    def print_points(img_building, key_points):
        img_building_keypoints = cv2.drawKeypoints(img_building,
                                                   key_points,
                                                   img_building,
                                                   flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  # Draw circles.
        plt.figure(figsize=(16, 16))
        plt.title('ORB Interest Points')
        plt.imshow(img_building_keypoints)
        plt.show()
        # input()

    def save_pictures_ransac(self, sorted_picture_list: List[Local_Picture], target_picture: Local_Picture, file_name):

        # TODO : Print transformation

        max_width = 0
        total_height = 0
        NB_BEST_PICTURES = 3

        # Preprocess to remove target picture from matches
        offset = json_class.remove_target_picture_from_matches(sorted_picture_list, target_picture)

        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            # We keep the largest picture
            max_width = max(target_picture.image.shape[1]*3 + sorted_picture_list[i + offset].image.shape[1]*3,max_width) # *2 for the skewed/deskewed
            # We keep the heighest picture
            total_height += max(target_picture.image.shape[0], sorted_picture_list[i + offset].image.shape[0])

        new_im = Image.new('RGB', (max_width, total_height))
        draw = ImageDraw.Draw(new_im)

        y_offset = 0
        for i in range(0, min(NB_BEST_PICTURES, len(sorted_picture_list))):
            trans_matrix_sorted_to_target = sorted_picture_list[i + offset].transformation_matrix
            trans_matrix_target_to_sorted = np.linalg.inv(sorted_picture_list[i + offset].transformation_matrix)

            # Affine version
            add_row = np.array([[0,0,1]])
            trans_matrix_sorted_to_target_affine = np.concatenate((sorted_picture_list[i + offset].transformation_rigid_matrix, add_row), axis=0)
            trans_matrix_target_to_sorted_affine = np.linalg.inv(np.concatenate((sorted_picture_list[i + offset].transformation_rigid_matrix, add_row), axis=0))

            # Get the size of the current matching picture
            h_target, w_target, d_target = target_picture.image.shape
            h_sorted, w_sorted, d_sorted = sorted_picture_list[i + offset].image.shape

            # Get the position of the 4 corners of the current matching picture
            pts_target = np.float32([[0, 0], [0, h_target - 1], [w_target - 1, h_target - 1], [w_target - 1, 0]]).reshape(-1, 1, 2)
            pts_sorted = np.float32([[0, 0], [0, h_sorted - 1], [w_sorted - 1, h_sorted - 1], [w_sorted - 1, 0]]).reshape(-1, 1, 2)

            try :
                # Transform the 4 corners thanks to the transformation matrix calculated
                dst_sorted_to_target = cv2.perspectiveTransform(pts_sorted, trans_matrix_sorted_to_target)
                dst_target_to_sorted = cv2.perspectiveTransform(pts_target, trans_matrix_target_to_sorted)

                # Affine version
                dst_sorted_to_target_affine = cv2.perspectiveTransform(pts_sorted, trans_matrix_sorted_to_target_affine)
                dst_target_to_sorted_affine = cv2.perspectiveTransform(pts_target, trans_matrix_target_to_sorted_affine)
                # OLD : dst_target_to_sorted_affine = cv2.warpAffine(pts_target, trans_matrix_target_to_sorted_affine, (w_target, h_target))

            except Exception as e :
                self.logger.error(f"Inverting RANSAC transformation matrix impossible due to : {e} on picture {sorted_picture_list[i + offset].path}. Is RANSAC the chosen filter ? {e}")
                self.logger.error(traceback.print_tb(e.__traceback__))
                continue

            # Draw the transformed 4 corners on the target picture (pic2, request)
            img_sorted = cv2.polylines( sorted_picture_list[i + offset].image.copy(), [np.int32( dst_target_to_sorted)], True, 255, 7, cv2.LINE_AA) # We draw the outline of the deskewed candidate over the target
            img_target = cv2.polylines( target_picture.image.copy(), [np.int32( dst_sorted_to_target)], True, 158, 7, cv2.LINE_AA) # We draw the outline of the target candidate over the deskewed

            # Affine version
            img_sorted_affine = cv2.polylines( img_sorted, [np.int32( dst_sorted_to_target_affine)], True, (100,0,100), 7, cv2.LINE_AA) # We draw the outline of the deskewed candidate over the target
            img_target_affine = cv2.polylines( img_target, [np.int32( dst_target_to_sorted_affine)], True, (100,0,100), 7, cv2.LINE_AA) # We draw the outline of the target candidate over the deskewed

            # Draw matches side to side between picture reference and picture request
            output = cv2.drawMatches(img_sorted_affine, sorted_picture_list[i + offset].key_points,
                                     img_target_affine, target_picture.key_points,
                                     sorted_picture_list[i + offset].matches, None)

            # Compute the transformation between picture reference and picture request (scale, and 3D angle)
            # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
            ss = trans_matrix_target_to_sorted[0, 1]
            sc = trans_matrix_target_to_sorted[0, 0]
            scaleRecovered = math.sqrt(ss * ss + sc * sc)
            thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
            self.logger.debug(f"MAP: Calculated scale difference: {scaleRecovered}, Calculated rotation difference: {thetaRecovered}" )

            # Deskew target picture (request picture) into a "transformed" version
            #skewed_picture = target_picture.image
            # orig_picture = sorted_picture_list[i + offset].image
            # Skewed picture // transformation matrix // size //
            im_out1 = cv2.warpPerspective(sorted_picture_list[i + offset].image, trans_matrix_sorted_to_target,(sorted_picture_list[i + offset].image.shape[1], sorted_picture_list[i + offset].image.shape[0]))
            im_out2 = cv2.warpPerspective(target_picture.image, trans_matrix_target_to_sorted,(target_picture.image.shape[1], target_picture.image.shape[0]))

            # Affine version
            im_out1_affine = cv2.warpPerspective(sorted_picture_list[i + offset].image, trans_matrix_sorted_to_target_affine,(sorted_picture_list[i + offset].image.shape[1], sorted_picture_list[i + offset].image.shape[0]))
            im_out2_affine = cv2.warpPerspective(target_picture.image, trans_matrix_target_to_sorted_affine,(target_picture.image.shape[1], target_picture.image.shape[0]))
            # OLD :  im_out2_affine = cv2.warpAffine(target_picture.image, trans_matrix_target_to_sorted_affine,(target_picture.image.shape[1], target_picture.image.shape[0]))

            # Copy paste the picture in the right global position of the output picture
            x_offset = 0
            new_im.paste(Image.fromarray(im_out2_affine), (x_offset , y_offset)) # width offset and height offset
            x_offset+= im_out2_affine.shape[1]
            new_im.paste(Image.fromarray(im_out2), (x_offset , y_offset)) # width offset and height offset
            x_offset+= im_out2.shape[1]
            new_im.paste(Image.fromarray(output), (x_offset, y_offset)) # height offset
            x_offset += output.shape[1]
            new_im.paste(Image.fromarray(im_out1), (x_offset, y_offset)) # width offset and height offset
            x_offset += im_out1.shape[1]
            new_im.paste(Image.fromarray(im_out1_affine), (x_offset, y_offset)) # width offset and height offset
            x_offset += im_out1_affine.shape[1]

            # Add title
            self.add_text_matches(i + offset, sorted_picture_list[i + offset], target_picture, y_offset, int(max_width*0.6), draw)

            # Set next offset
            y_offset += Image.fromarray(output).size[1]

        self.logger.debug(f"Save to : {str(file_name)}")
        print(file_name)

        new_im.save(file_name)

    def add_text_matches(self, picture_index, sorted_curr_picture, target_picture, y_offset, max_width, draw):

        # Print nice text
        P1 = "LEFT = BEST MATCH #" + str(picture_index) + " d=" + str(sorted_curr_picture.distance)
        P2 = " at " + sorted_curr_picture.path.name
        P3 = "| RIGHT = ORIGINAL IMAGE"
        P4 = " at " + target_picture.path.name + "\n"

        if sorted_curr_picture.description is not None:
            P5 = str(len(sorted_curr_picture.description)) + " descriptors for LEFT "
        else:
            P5 = "NONE DESCRIPTORS LEFT "

        if target_picture.description is not None:
            P6 = str(len(target_picture.description)) + " descriptors for RIGHT "
        else:
            P6 = "NONE DESCRIPTORS RIGHT "

        if sorted_curr_picture.matches is not None:
            P7 = str(len(sorted_curr_picture.matches)) + "# matches "
        else:
            P7 = "NONE MATCHES "

        tmp_title = P1 + P2 + P3 + P4 + P5 + P6 + P7
        self.text_and_outline(draw, 10, y_offset + 10, tmp_title, font_size=max_width // 60)

        return draw
