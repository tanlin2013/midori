import cv2, pyautogui
import numpy as np


class OpencvWrapper:
    
    @staticmethod
    def screenshot(coordinate=None):
        """
        Args:
            coordinate: (x1, y1, x2, y2) with (x1, y1) in 2nd Quadrant and (x2, y2) in 4th Quadrant.
        
        Returns:
            img (Image):
        """
        if coordinate:
            x, y = coordinate[:2]
            width = coordinate[2] - x
            height = coordinate[3] - y
            print(x, y, width, height)
            img = pyautogui.screenshot(region=(x, y, width, height))
        else:
            img = pyautogui.screenshot()
        return img
    
    @staticmethod
    def image_gray(img_rgb):
        return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def search_image(background, template_path, precision=0.95):
        img_rgb = np.array(background)
        img_gray = OpencvWrapper.image_gray(img_rgb)
        template_rgb = cv2.imread(template_path)
        if template_rgb is None: raise TypeError("Image is not found in path: {}".format(template_path))
        template_gray = OpencvWrapper.image_gray(template_rgb)
        
        size = np.asarray(template_gray.shape[::-1])
        #  TODO: https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate#matchtemplate
        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
#        loc = np.where(res >= precision)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#        coordinate = []
#        for pt in zip(*loc[::-1]):
#            coordinate.append([pt[0], pt[1], pt[0]+width, pt[1]+height])
        top_left = np.add(np.asarray(max_loc), -0.5*size)
        bottom_right = np.add(np.asarray(max_loc), 0.5*size)
        coordinate = [top_left, bottom_right]
        return coordinate, res
    
    @staticmethod
    def click_image():
        return
    
# =============================================================================
# if __name__ == '__main__':
#     
#     coor, res = OpencvWrapper.search_image(pyautogui.screenshot(), '../envs/test.png')
#     print(coor)
#     print(np.linalg.norm(res))
# =============================================================================
    
    