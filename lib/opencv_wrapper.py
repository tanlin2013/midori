import cv2, pyautogui
import numpy as np

class OpencvWrapper:
    
    @staticmethod
    def screenshot(coordinate):
        """
        @param coordinate: (x1, y1, x2, y2) with (x1, y1) in 3rd Quadrant and (x2, y2) in 1st Quadrant.
        
        @returns img (Image):
        """
        x, y = coordinate[:2]
        width = coordinate[2] - x
        height = coordinate[3] - y
        return pyautogui.screenshot(region=(x, y, width, height))
    
    @staticmethod
    def image_gray(img_rgb):
        return cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def search_image(background, template_path, precision=0.9):
        img_rgb = np.array(background)
        img_gray = OpencvWrapper.image_gray(img_rgb)
        template_rgb = cv2.imread(template_path)
        template_gray = OpencvWrapper.image_gray(template_rgb)
        width, height = template_gray.shape[::-1]

        res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= precision)
        coordinate = []
        for pt in zip(*loc[::-1]):
            coordinate.append((pt[0], pt[1], pt[0]+width, pt[1]+height))
        return coordinate
    
    @staticmethod
    def click_image():
        return
    
# =============================================================================
# if __name__ == '__main__':
#     
#     coor = OpencvWrapper.search_image(pyautogui.screenshot(), './test.png')
#     print(coor)
# =============================================================================
