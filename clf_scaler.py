import time
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from skimage.feature import hog
import cv2
import glob
from tqdm import tqdm
import pickle


def hist_feature(img, bins, channels):
    '''
    Returns a vector of histogram features for channels `channels` of an image
    and with `bins` bins.
    '''
    hist_vec = []

    # if more than 1 channel, concatenate the features together
    if len(channels) > 1:
        for ch in channels:
            vec, _ = np.histogram(img[:, :, ch].ravel(), bins=bins, normed=True)
            hist_vec.extend(vec)
    else:
        hist_vec = np.histogram(img[:, :, channels[0]], bins=bins, normed=True)
    return hist_vec

def hog_feature(img, channels):
    '''
    Returns a vector of HOG features of an image `image`.
    channels - (list) list of channels to extract the features from.
    '''
    hog_vec = []
    if len(channels) > 1:
        for ch in channels:
            vec = hog(img[:, :, ch], orientations=8, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), block_norm='L2-Hys',
                      visualise=False, transform_sqrt=True, feature_vector=True)
            hog_vec.extend(vec)
    else:
        hog_vec = hog(img[:, :, channels[0]], orientations=8, pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2), block_norm='L2-Hys',
                  visualise=False, transform_sqrt=True, feature_vector=True)
    return hog_vec

def feature_vec(img_list, hist_bins, channels):
    '''
    Return a list of vectors made of hog and histogram features for
    the images from the `img_list`
    '''
    fvec = []
    for img in tqdm(img_list, desc="feature_vec"):
        # include all three channels in histogram feature extraction
        hist_vec = hist_feature(img, hist_bins, [0, 1, 2])
        hog_vec = hog_feature(img, channels)
        fvec.append(np.concatenate((hog_vec, hist_vec)))
    return fvec

def open_transform(filenames, colorspace, size):
    '''
    Returns a list of images from list `filenames`,
    transformed to `colorspace` and resized to `size`.
    '''
    images = []
    cvt_color_str = 'cv2.COLOR_BGR2' + colorspace
    for file in tqdm(filenames, desc="open_transform"):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, eval(cvt_color_str))
        if img.shape[:2] != size:
            img = cv2.resize(img, size)
        images.append(img)
    return images

# lists all trainig images
veh_files = glob.glob('/data/vehicles/*/*')
nonveh_files = glob.glob('/data/non-vehicles/*/*')
# assign label '1' to vehicles and '0' to non-vehicles
all_labels = np.hstack((np.ones(len(veh_files)), np.zeros(len(nonveh_files)))).astype(np.int8)
all_veh = veh_files
all_veh.extend(nonveh_files)
all_veh = np.array(all_veh)
# use stratified shuffle split to improve randomization of the train data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
train_idx, test_idx = next(sss.split(all_veh, all_labels))

color_space = 'HSV' # color space used
channels = [1, 2] # channels used by the classifier
hist_bins = 64 # number of histogram bins
windows = [(8, 8), (8, 12)] # sizes of windows (cell size, number of cells)

# used to store classifiers and their scalers
clf_scaler = []

# iterate through the list of windows sizes
for i, (cell_size, number_of_cells) in enumerate(windows):
    # initialize classifier
    clf = LinearSVC()
    # initialize scaler
    scaler = StandardScaler()

    # resize the image to desired size
    new_size = (cell_size*number_of_cells, cell_size*number_of_cells)
    # opens and transforms the image from the training set
    images = open_transform(all_veh[train_idx], 'HSV', new_size)
    # converting images into features vectors
    feature_vec_list = feature_vec(images, hist_bins, channels)
    # fitting the scaler and transforming features vectors
    X = scaler.fit_transform(feature_vec_list)
    y = all_labels[train_idx]
    print("scaler fitted.")
    # fitting the classifier
    clf.fit(X, y)
    print("fitting classifier finished.")
    # adding classifier and scaler to the list
    clf_scaler.append(dict(clf=clf, scaler=scaler))

    # testing
    test_images = open_transform(all_veh[test_idx], 'HSV', new_size)
    test_vec = feature_vec(test_images, hist_bins, channels)
    test_vec_trans = scaler.transform(test_vec)
    print("window {}, score: {:.2f}".format(i, 100*clf.score(test_vec_trans, all_labels[test_idx])))

# saving list of classifiers and scalers
pickle.dump(clf_scaler, open("/output/clf_scaler.p", "wb"))
