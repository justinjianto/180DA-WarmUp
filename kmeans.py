"""
Justin Jianto
Sources:
https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
https://stackoverflow.com/questions/58887056/resize-frame-of-cv2-videocapture
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0)
 
while 1:
    ret,frame =cap.read()
    # ret will return a true value if the frame exists otherwise False
    if ret == True:
        frame = cv2.resize(frame, (200, 200))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1],3))
        clt = KMeans(n_clusters=4)
        clt.fit(img)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)

        bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

        cv2.imshow('feed', frame)
        cv2.imshow('dominant color', bar)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
 
cv2.destroyAllWindows()