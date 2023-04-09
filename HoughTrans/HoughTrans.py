import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

FigPath = '../dataset'
SavePath = './results'

def Hough_detect(image_h, edges, rho, theta, theta_range, threshold, path, name):
    min_theta = 0
    max_theta = min_theta + theta_range
    lines = cv.HoughLines(edges, rho, theta, threshold, min_theta = min_theta, max_theta = max_theta)
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv.line(image_h, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv.imwrite(os.path.join(path, name), image_h)
    return len(lines)

def PHough_detect():
    pass

if not os.path.exists(SavePath):
    os.mkdir(SavePath)

for root, dirs, files in os.walk(FigPath, topdown=False):
    for name in files:
        path = os.path.join(root, name)
        image_h = cv.imread(path)
        image_p = cv.imread(path)
        image_bw = cv.imread(path, 0)
        edges = cv.Canny(image_bw, 50, 200)
        ## Employ Hough transform
        # rho = 1
        # theta = np.pi/180
        # threshold = 200 
        # theta_range = np.pi/2
        # lines_num = Hough_detect(image_h, edges, rho, theta, theta_range, threshold, path = SavePath, name = f'{rho}_{round(theta, 3)}_{theta_range}_{threshold}_{name}')
        for rho in [1, 10, 100, 1000, 10000]:
            pass
        for theta in [np.pi/45, np.pi/90, np.pi/180, np.pi/360, np.pi/720]:
            pass
        for theta_range in [1, 0.8, 0.5, 0.3, 0.1]:
            pass
        for threshold in [50, 100, 200, 300, 1000]:
            lines_num = Hough_detect(image_h.copy(), edges, rho, theta, theta_range * np.pi, threshold, \
                path = SavePath, name = f'{rho}_{theta_range}PI_{threshold}_{name}')
                        

        ## Employ 

