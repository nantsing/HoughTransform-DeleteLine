import cv2 as cv
import numpy as np
from sklearn.cluster import MeanShift, KMeans

# Canny binarization
image_raw = cv.imread('./1.png')
image_bw = cv.imread('./1.png', 0)
edges = cv.Canny(image_bw, 50, 200)
cv.imwrite('./results/1.png', edges)

# Hough transform ==> horizontal line
Y = []
image = image_raw.copy()
lines = cv.HoughLines(edges, 3, np.pi / 180, 50, min_theta = np.pi / 2, max_theta = np.pi / 2)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a)) + 1
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a)) + 1
    Y.append(y1)
    cv.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv.imwrite('./results/2.png', image)

# Clustering (Kmeans)
Y = np.array(Y).reshape(-1, 1)
clustering = KMeans(n_clusters = 11).fit(Y)
delete_line = np.floor(clustering.cluster_centers_.reshape(1, -1)[0])
x1 = 0
x2 = 1000
image = image_raw.copy()
for y in delete_line:
    cv.line(image, (x1, int(y)), (x2, int(y)), (0, 0, 255), 2)
cv.imwrite('./results/3.png', image)

# Determine line segment
X = []
m, n = edges.shape
for y in delete_line:
    x = 0
    x1 = -1
    x2 = -1
    while x1 == -1 and edges[int(y), x] == 0: x = x + 1
    x1 = x - 5
    x = n - 1
    while x2 == -1 and edges[int(y), x] == 0: x = x - 1
    x2 = x + 5
    X.append([x1, x2])
    
# Draw lines
image = image_raw.copy()
for xx, y in zip (X, delete_line):
    x1, x2 = xx
    cv.line(image, (x1, int(y)), (x2, int(y)), (0, 0, 255), 2)
cv.imwrite('./results/4.png', image)