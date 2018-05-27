import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from skimage.feature import hog
import cv2
import glob
from tqdm import tqdm
import pickle

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

def hog_feature(img, channels):
    '''
    Returns a vector of hog features of images' channels
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
    Combines hog and histogram feature vectors into a list of
    features vectors and returns it
    '''
    fvec = []
    for img in tqdm(img_list, desc="feature_vec"):
        hist_vec = hist_feature(img, hist_bins, [0, 1, 2])
        hog_vec = hog_feature(img, channels)
        fvec.append(np.concatenate((hog_vec, hist_vec)))
    return fvec

def open_transform(filenames, indexes, colorspace, size):
    '''
    Opens filenames[indexes], converts them to new color space,
    resizes them and retuns a list of them
    '''
    images = []
    cvt_color_str = 'cv2.COLOR_BGR2' + colorspace
    for idx in tqdm(indexes, desc="open_transform"):
        file = filenames[idx]
        img = cv2.imread(file)
        img = cv2.cvtColor(img, eval(cvt_color_str))
        if img.shape[:2] != size:
            img = cv2.resize(img, size)
        # every filename is a duplicate of the previous one
        # flips every odd image to augment the data
        if idx % 2 == 0:
            images.append(img)
        else:
            images.append(np.fliplr(img))
    return images

if __name__ == '__main__':
    veh_files = glob.glob('data/vehicles/*/*')
    # duplicate every element of `veh_files` list
    veh_files = [file for file in veh_files for _ in (0, 1)]
    nonveh_files = glob.glob('data/non-vehicles/*/*')
    # duplicate every element of `nonveh_files` list
    nonveh_files = [file for file in nonveh_files for _ in (0, 1)]
    # labels: 1 - vehicle, 0 - non-vehicle
    all_labels = np.hstack((np.ones(len(veh_files)), np.zeros(len(nonveh_files)))).astype(np.int8)
    all_veh = veh_files
    all_veh.extend(nonveh_files)
    all_veh = np.array(all_veh)
    # use stratified shuffle split to better randomize the training data
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    train_idx, test_idx = next(sss.split(all_veh, all_labels))

    ###########################
    # Hyperparameters         #
    ###########################
    color_space = 'HSV'
    channels = [1, 2]
    hist_bins = 32
    n_estimators = 4
    windows = [(8, 3), (8, 4), (8, 5)]
    ###########################

    clf_scaler = []

    for i, (cell_size, number_of_cells) in enumerate(windows):
        # some of the classifiers I've tried:
        #clf = SVC(verbose=3)
        #clf = GaussianNB()
        #clf = OneVsRestClassifier(BaggingClassifier(SVC(),
        #                                            max_samples=1.0 / n_estimators,
        #                           n_estimators=n_estimators, n_jobs=-1, verbose=0))
        #clf = BaggingClassifier(SVC(), max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1, verbose=0)


        clf = LinearSVC()
        scaler = StandardScaler()

        new_size = (cell_size*number_of_cells, cell_size*number_of_cells)
        images = open_transform(all_veh, train_idx, color_space, new_size)
        feature_vec_list = feature_vec(images, hist_bins, channels)

        # fit the scaler and transform the training data
        X = scaler.fit_transform(feature_vec_list)
        y = all_labels[train_idx]

        # fit the classifier
        clf.fit(X, y)

        # add classifer and the scaler to the list
        clf_scaler.append(dict(clf=clf, scaler=scaler))

        # verify
        test_images = open_transform(all_veh, test_idx, color_space, new_size)
        test_vec = feature_vec(test_images, hist_bins, channels)
        test_vec_trans = scaler.transform(test_vec)
        print("window {}, score: {:.2f}".format(i, 100*clf.score(test_vec_trans, all_labels[test_idx])))

    # add hyperparameters to the list
    clf_scaler.append(dict(windows=windows, cspace=color_space, bins=hist_bins, channels=channels))
    # dump the pickle
    pickle.dump(clf_scaler, open("/output/clf_scaler.p", "wb"))
