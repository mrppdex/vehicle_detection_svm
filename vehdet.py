import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from skimage.feature import hog
from scipy.ndimage.measurements import label
from collections import deque
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

def add_heat(heatmap, bboxes):
    '''
    Returns the heat map updated with the list of `bboxes`
    '''
    heat = np.zeros_like(heatmap)
    for bbox in bboxes:
        heat[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]] += 1
    return heat


def process_image(image):
    '''
    Returns an input image with vehicles bounded by the boxes
    '''
    global vertices
    global multi_frame_boxes

    # region of interest of the original image
    hsv_roi = image[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0], :]
    # convert from RGB to the desired color space
    hsv_roi = cv2.cvtColor(hsv_roi, eval("cv2.COLOR_RGB2" + color_space))

    # resize the slice of the image by multiplying its dimensions by `scale`
    new_roi_size = np.array(hsv_roi.shape[:2][::-1])*scale
    hsv_roi = cv2.resize(hsv_roi, tuple(new_roi_size.astype(np.int)))

    # initialize the heat map
    heat_map = np.zeros(hsv_roi.shape[:2])

    # calculate hog map for the s channel of the resized slice of the image
    hog_s = hog(hsv_roi[:, :, 1],
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualise=False,
                transform_sqrt=True,
                feature_vector=False)

    # calculate hog map for the v channel of the resized slice of the image
    hog_v = hog(hsv_roi[:, :, 2],
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm='L2-Hys',
                visualise=False,
                transform_sqrt=True,
                feature_vector=False)

    boxes = [] # used to store all detected boxes in an image

    # first and last cell on the y axis to sweep through
    # for each of the classifier windows
    start_stop = [(0, 2), (0, 7), (5, 10)]

    # iterate through all the classifier windows
    for i, win in enumerate(windows):

        # load classifier
        clf = clf_scaler[i]['clf']
        # load scaler
        scaler = clf_scaler[i]['scaler']

        # slide the window in the designated areas
        for win_y in range(*start_stop[i]):
            for win_x in range(nwin_x[i]+1):
                xl = win_x * steps[i][0] # x left cell
                xr = xl + win[1] - 1  # x right cell
                yu = win_y * steps[i][1] # y upper cell
                yd = yu + win[1] - 1# y lower cell

                # hog map features
                win_s = hog_s[yu:yd, xl:xr]
                win_v = hog_v[yu:yd, xl:xr]

                # histogram features
                hist_img = hsv_roi[yu*win[0]:(yd+1)*win[0], xl*win[0]:(xr+1)*win[0]]
                hist_vec = np.array(hist_feature(hist_img, hist_bins, [0, 1, 2]))

                # concatenate hog and histogram features into one vector
                win_feature = np.hstack((win_s.ravel(), win_v.ravel(), hist_vec)).reshape(1, -1)
                # scale features vector
                win_feature = scaler.transform(win_feature)

                # if the current window contain a vehicle update the heat map
                if clf.predict(win_feature) == 1:
                    bbox = ((yu*win[0],xl*win[0]),((yd+1)*win[0],(xr+1)*win[0]))
                    # add box with predicted vehicle to the boxes list
                    boxes.append(bbox)

    # collect all the boxes within the last 6 frames
    multi_frame_boxes.append(boxes)

    # update the heatmap
    heat_map = add_heat(heat_map , [bbox for bboxes in multi_frame_boxes for bbox in bboxes])
    # thresholding
    heat_map[heat_map < 5] = 0
    # labelize
    labels = label(heat_map)

    v_candidate = [] # used to store all boxes candidates

    # iterate through all discovered labels
    for car_number in range(1, labels[1]+1):

        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0]) # + roi[0][1]
        nonzerox = np.array(nonzero[1]) # + roi[0][0]
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Translate it to the original image size
        bbox_trans = ((np.min(nonzerox/scale+roi[0][0]).astype(np.int),
                       np.min(nonzeroy/scale+roi[0][1]).astype(np.int)),
                     (np.max(nonzerox/scale+roi[0][0]).astype(np.int),
                      np.max(nonzeroy/scale+roi[0][1]).astype(np.int)))

        # Ignore boxes with area smaller than 1000
        if (bbox_trans[1][0]-bbox_trans[0][0])*(bbox_trans[1][1]-bbox_trans[0][1]) < 1000:
            continue

        # Check if upper left-corner is within 120px radius
        # of any of the upper-left corners from the previous frame
        max_distance=120
        is_close = False # set to true if the bounding box is already drawn

        # go through the list of all current bounding bboxes
        # if new box is within the `max_distance` radius of another previous box
        # then remove the old one and replace it with the new one
        for vert in vertices:
            # v - top-left corner of the bounding box
            # b - bounding box translated to the original image size
            # c - counts how many times bounding box was predicted to be a vehicle
            v, b, c = vert

            # if bounding box has not been updated for a while, remove it
            if c < 0:
                continue

            # update the position of a bounding box
            if not is_close and sq_distance(bbox[0], v) <= (max_distance*scale)**2:
                if c > 8:
                    cv2.rectangle(image, bbox_trans[0], bbox_trans[1], (0, 255, 0), 6)
                v_candidate.append((bbox[0], bbox_trans, c+1))
                vertices.remove(vert)
                is_close = True
                continue

            # update reference for previous, not updated bounding boxes
            v_candidate.append((v, b, c-1))
            vertices.remove(vert)
            # if bounding box has enough references, draw it
            if c > 8:
                cv2.rectangle(image, *b, (0, 255, 0), 6)

        # add newly discovered bounding box to list
        v_candidate.append((bbox[0], bbox_trans, 0))

        # replace the global vertices list
        vertices = v_candidate if v_candidate else vertices

    return image


if __name__ == '__main__':
    # load the pickle
    clf_scaler = pickle.load(open("clf_scaler.p", "rb"))

    windows = clf_scaler[-1]['windows'] # list of classifer windows trained
    color_space = clf_scaler[-1]['cspace'] # color space
    hist_bins = clf_scaler[-1]['bins'] # number of histogram bins
    channels = clf_scaler[-1]['channels'] # color spaces channels

    # steps (in cells) in x and y direction for every used classifier window
    steps = [(2, 1), (2, 1), (2, 1)]
    roi = ((675, 400), (1280, 680)) # region of interest (cv notation)
    scale = 0.4 # scale of the original image size

    # span in x and y direction in resized roi
    span_x = int((roi[1][0] - roi[0][0])*scale)
    span_y = int((roi[1][1] - roi[0][1])*scale)

    ncells_x, ncells_y = [], [] # number of cells in roi for every window size
    nwin_x, nwin_y = [], [] # number of windows in roi for every window size

    for i, win in enumerate(windows):
        ncells_x.append(span_x // win[0])
        ncells_y.append(span_y // win[0])
        nwin_x.append((ncells_x[i] - win[1])//steps[i][0])
        nwin_y.append((ncells_y[i] - win[1])//steps[i][1])

    vertices = [] # used to keep all current bounding boxes
    multi_frame_boxes = deque(maxlen=6) # used to update the heat map

    vid_output = 'output/vehicle_detection.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    vid_clip = clip1.fl_image(process_image)
    vid_clip.write_videofile(vid_output, audio=False)
