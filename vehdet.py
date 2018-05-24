import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
from scipy.ndimage.measurements import label
import glob
import pickle
from tqdm import tqdm
from moviepy.editor import VideoFileClip
import imageio
import cv2

imageio.plugins.ffmpeg.download()


def hist_feature(img, bins, channels):
    '''
    Returns a vector of histogram features for `channels` of an image
    and with `bins` number of bins.
    '''
    hist_vec = []
    if len(channels) > 1:
        for ch in channels:
            vec, _ = np.histogram(img[:, :, ch].ravel(), bins=bins, normed=True)
            hist_vec.extend(vec)
    else:
        hist_vec = np.histogram(img[:, :, channels[0]], bins=bins, normed=True)
    return hist_vec

def sq_distance(p1, p2):
    '''
    Returns squared distance between points p1 and p2
    '''
    return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2

def process_image(image):
    '''
    Returns an image with bounding boxes around detected vehicles.
    '''
    # store vertices in a global variable
    global vertices

    # convert image to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cut out the region of interest
    hsv_roi = hsv_img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0], :]
    # initialize heat map
    heat_map = np.zeros(hsv_roi.shape[:2])
    # calculate hog map for s channel of roi
    hog_s = hog(hsv_roi[:, :, 1],
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualise=False,
                transform_sqrt=True,
                feature_vector=False)
    # calculate hog map for v channel of roi
    hog_v = hog(hsv_roi[:, :, 2],
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualise=False,
                transform_sqrt=True,
                feature_vector=False)

    # iterate through different sizes of windows
    for i, win in enumerate(windows):
        # read classifier and scaler
        clf = clf_scaler[i]['clf']
        scaler = clf_scaler[i]['scaler']

        # slide window in y direction
        for win_y in range(nwin_y[i]):
            # slide window in x direction
            for win_x in range(nwin_x[i]):
                # corners of the window (in cells)
                xl = win_x * steps[i][0] # x left corner
                xr = xl + win[1] - 1  # x right corner
                yu = win_y * steps[i][1] # y upper corner
                yd = yu + win[1] - 1 # y lower corner
                # extract hog features from the maps
                win_s = hog_s[yu:yd, xl:xr]
                win_v = hog_v[yu:yd, xl:xr]
                # translate cells to pixels and calculate histogram for the region
                hist_img = hsv_roi[yu*win[0]:yd*win[0], xl*win[0]:xr*win[0]]
                hist_vec = np.array(hist_feature(hist_img, hist_bins, [0, 1, 2]))
                # concatenate feature vectors into one vector
                win_feature = np.hstack((win_s.ravel(), win_v.ravel(), hist_vec)).reshape(1, -1)
                # scale it
                win_feature = scaler.transform(win_feature)

                # if detected a vehicle, update the heat map
                if clf.predict(win_feature) == 1:
                    heat_map[yu*win[0]:(yd+1)*win[0], xl*win[0]:(xr+1)*win[0]] += 1

    # scale the heat map to range 0..9
    # keep values lower than the mean value closer to 0
    # keep values greater than the mean value closer to 9
    hmap = (((heat_map/np.max(heat_map)+0.5)**2)*4).astype(np.uint8)
    # remove values lower than 5
    hmap[hmap < 5] = 0

    # block used to plot heatmap on the final image
    im = image[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0], :]
    im[hmap > 0] = (im[hmap > 0]*0.5).astype(np.uint8)
    im[hmap > 0, 0] = 0
    im[:, :, 0] += ((hmap/np.max(hmap))*255.).astype(np.uint8)

    # get labels
    labels = label(hmap)

    v_candidate = [] # bounding boxes candidates

    # iterate through all labels
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0]) + roi[0][1]
        nonzerox = np.array(nonzero[1]) + roi[0][0]
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Ignore boxes with area smaller than 2000
        if (bbox[1][0]-bbox[0][0])*(bbox[1][1]-bbox[0][1]) < 2000:
            continue

        # Check if upper left-corner is within 100px radius
        # of any of the upper-left corners from the previous frame
        if vertices:
            for v in vertices:
                if sq_distance(bbox[0], v) <= 10000: #100^2
                    cv2.rectangle(image, bbox[0], bbox[1], (255, 0, 0), 6)
                    v_candidate.append(bbox[0])
                    break
                else:
                    v_candidate.append(bbox[0])
        else:
            cv2.rectangle(image, bbox[0], bbox[1], (255, 0, 0), 6)
            vertices.append(bbox[0])
            continue

    vertices = v_candidate
    return image

if __name__ == '__main__':
    # load the pickle
    clf_scaler = pickle.load(open("clf_scaler_new.p", "rb"))

    color_space = 'HSV' # color space used
    channels = [1, 2] # channels used to extract hog features
    hist_bins = 64 # number of histogram bins
    windows = [(8, 8), (8, 12)] # cell size (pixels) x number of cells
    steps = [(4, 6), (2, 2)] # steps (in cells) for corresponding windows
    roi = ((600, 370), (1280, 680)) # in cv notation

    span_x = roi[1][0] - roi[0][0] # span of roi along x axis
    span_y = roi[1][1] - roi[0][1] # span of roi along y axis

    ncells_x, ncells_y = [], [] # number of cells in roi for every window size
    nwin_x, nwin_y = [], [] # number of windows in roi for every window size

    for i, win in enumerate(windows):
        ncells_x.append(span_x // win[0])
        ncells_y.append(span_y // win[0])
        nwin_x.append((ncells_x[i] - win[1])//steps[i][0])
        nwin_y.append((ncells_y[i] - win[1])//steps[i][1])

    vertices = []
    vid_output = 'output/project_detection.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    vid_clip = clip1.fl_image(process_image)
    vid_clip.write_videofile(vid_output, audio=False)
