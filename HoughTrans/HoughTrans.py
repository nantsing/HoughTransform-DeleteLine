import os
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

FigPath = '../dataset'
SavePath = './results'

def Hough_detect(image_h, edges, rho, theta, theta_range, threshold, path, name):
    min_theta = 0
    max_theta = min_theta + theta_range
    lines = cv.HoughLines(edges, rho, theta, threshold, min_theta = min_theta, max_theta = max_theta)
    if lines is None: lines = []
    name = f'{len(lines)}_' + name
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

def PHough_detect(image_p, edges, rho, theta, threshold, minLineLength, maxLineGap, path, name):
    lines = cv.HoughLinesP(edges, rho, theta, threshold, \
        minLineLength = minLineLength, maxLineGap = maxLineGap)
    if lines is None: lines = []
    name = f'PHough_{len(lines)}_' + name
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line( image_p, (x1, y1), (x2, y2), (0, 255, 0), 2 )
    cv.imwrite(os.path.join(path, name), image_p)
    return len(lines)

def plot_bar(lines_num_list, y_label, x_value, x_label, figname, SavePath):
    plt.figure()
    print(lines_num_list)
    plt.bar(x_value, lines_num_list, alpha = 0.6)
    plt.grid(True, linestyle = ':', color= 'r' , alpha = 0.6)
    
    for a,b in zip(x_value, lines_num_list):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10);
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{figname}')
    plt.savefig(f'{SavePath}/{figname}.png')

if __name__ == '__main__':

    if not os.path.exists(SavePath):
        os.mkdir(SavePath)

    for root, dirs, files in os.walk(FigPath, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            image_h = cv.imread(path)
            image_p = cv.imread(path)
            image_bw = cv.imread(path, 0)
            edges = cv.Canny(image_bw, 50, 300)
            ## Employ Hough transform
            rho = 1
            radian = np.pi/180
            theta_res = 1
            threshold = 200 
            theta_range = 1
            lines_num_list = []
            x_value = []
            # lines_num = Hough_detect(image_h, edges, rho, theta, theta_range, threshold, path = SavePath, name = f'{rho}_{round(theta, 3)}_{theta_range}_{threshold}_{name}')
            for rho in [1, 10, 100, 1000, 10000]:
                theta = radian / theta_res
                lines_num = Hough_detect(image_h.copy(), edges, rho, theta, theta_range * np.pi, threshold, \
                    path = SavePath, name = f'{rho}_{theta_res}_{theta_range}PI_{threshold}_{name}')
                lines_num_list.append(lines_num)
                x_value.append(str(rho))
            plot_bar(lines_num_list, 'Number of lines', x_value, 'rho resolution', 'different rho resolution - number of lines', SavePath)
            lines_num_list = []
            x_value = []
            rho = 1
                
            for theta_res in [0.25, 0.5, 1, 2, 4]:
                theta = radian / theta_res
                lines_num = Hough_detect(image_h.copy(), edges, rho, theta, theta_range * np.pi, threshold, \
                    path = SavePath, name = f'{rho}_{theta_res}_{theta_range}PI_{threshold}_{name}')
                lines_num_list.append(lines_num)
                x_value.append(str(theta_res))
            plot_bar(lines_num_list, 'Number of lines', x_value, 'theta resolution', 'different theta resolution - number of lines', SavePath)
            lines_num_list = []
            x_value = []
            theta_res = 1
                
            for theta_range in [1, 0.8, 0.5, 0.3, 0.1]:
                theta = radian / theta_res
                lines_num = Hough_detect(image_h.copy(), edges, rho, theta, theta_range * np.pi, threshold, \
                    path = SavePath, name = f'{rho}_{theta_res}_{theta_range}PI_{threshold}_{name}')
                lines_num_list.append(lines_num)
                x_value.append(str(theta_range) + 'PI')
            plot_bar(lines_num_list, 'Number of lines', x_value, 'theta range', 'different theta range - number of lines', SavePath)
            lines_num_list = []
            x_value = []
            theta_range = 1
            
            for threshold in [50, 100, 200, 300, 1000]:
                theta = radian / theta_res
                lines_num = Hough_detect(image_h.copy(), edges, rho, theta, theta_range * np.pi, threshold, \
                    path = SavePath, name = f'{rho}_{theta_res}_{theta_range}PI_{threshold}_{name}')
                lines_num_list.append(lines_num)
                x_value.append(str(threshold))
            plot_bar(lines_num_list, 'Number of lines', x_value, 'theta resolution', 'different threshold - number of lines', SavePath)
            lines_num_list = []
            x_value = []
            threshold = 200

            minLineLength = 100
            maxLineGap = 10
            ## Employ Probabilistic Hough transform
            for minLineLength in [1, 10, 50, 100, 200]:
                theta = radian / theta_res
                lines_num = PHough_detect(image_p.copy(), edges, rho, theta, threshold, minLineLength, maxLineGap, \
                    path = SavePath, name = f'{rho}_{theta_res}_{threshold}_{minLineLength}_{maxLineGap}_{name}')
                lines_num_list.append(lines_num)
                x_value.append(str(minLineLength))
            plot_bar(lines_num_list, 'Number of lines', x_value, 'minLineLength', 'different minLineLength - number of line segments', SavePath)
            lines_num_list = []
            x_value = []
            minLineLength = 100
            
            for maxLineGap in [1, 10, 50, 100, 200]:
                theta = radian / theta_res
                lines_num = PHough_detect(image_p.copy(), edges, rho, theta, threshold, minLineLength, maxLineGap, \
                    path = SavePath, name = f'Phough_{rho}_{theta_res}_{threshold}_{minLineLength}_{maxLineGap}_{name}')
                lines_num_list.append(lines_num)
                x_value.append(str(maxLineGap))
            plot_bar(lines_num_list, 'Number of lines', x_value, 'maxLineGap', 'different maxLineGap - number of line segments', SavePath)
            lines_num_list = []
            x_value = []
            maxLineGap = 10
            
            