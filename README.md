|File|Description|
|-|-|
|clf_scaler.py| Fits classifiers and scalers. Saves a pickle with them to `clf_scaler.p`|
|vehdet.py| Processes the frames. Uses fitted `clf_scaler.p`|
|clf_scaler.p| Pickle with a list of {classifier, scaler} pairs|


# Vehicle Detection using SVM Classifier
*Author:* **Pawel Piela**
*Date:* **15/5/2018**

[test_bounded]: output_images/test_bounded.png
[test_roi]: output_images/test_roi.png
[hogmap]: output_images/hogmap.png
[heat_pre]: output_images/heatmap_pre.png
[heat_post]: output_images/heatmap_post.png
[label_map]: output_images/label_map.png
[label_boxes]: output_images/label_boxes.png
[o1]: output_images/output1.png
[o2]: output_images/output2.png
[o3]: output_images/output3.png
[sliding]: output_images/sliding_win.png

## Project

The goal of this project is to write a vehicle detection pipeline, using Support Vector Machine (SVM) Classifier, which consists of the following steps:

- Convert color space,
- Calculate HOG features,
- Calculate Histogram features,
- Fit the Classifier and the Scaler,
- Define the Region of Interest,
- Implement sliding window technique,
- Predict the content of a window (vehicle or non-vehicle),
- Update the heat map,
- Draw rectangles around detected vehicles,

## Convert color space.

I use Mohan Khartik's hyperparameters from [his blog on Medium](https://medium.com/@mohankarthik/feature-extraction-for-vehicle-detection-using-hog-d99354a84d10). I had built a cross validation program that uses `GridSearchCV` function from `sklearn` library to find optimal parameters, but every step of cross validation requires image transformation, fitting scaler and classifier. It's more that my laptop can handle in a reasonable amount of time. It requires a lot of processing speed of the CPU. GPU is not used to fit LinearSVC classifier.

I'm converting images to HSV Color Space, use S and V layers to extract HOG features and ALL layers to extract histogram features for the region of interest.

## Calculate HOG features.

HOG Hyperparameters:

|Name|Value|
|-|-|
|Orientations|8|
|Pixels Per Cell| (8, 8)|
|Cells Per Block| (2, 2)|
|`transform_sqrt`| `TRUE`|

I calculate the HOG map for the whole ROI, so I can use it later in my sliding window implementation. The visualisation for the test image looks as follows:

![HOG map][hogmap]

## Calculate Histogram features.

I use all three channels of HSV image and `number of bins` set to `64`. I also use histogram normalization.

```python
vec, _ = np.histogram(img[:, :, ch].ravel(), bins=bins, normed=True)
```

## Fit the Classifier and the Scaler.

I use Linear Support Vector Machine Classifier `LinearSVC` from the `sklearn` library. I use default parameters (`C=1.0`). I have also tested `GaussianNB` and `SVC` with `rbf` kernel. I got the following scores:

|Classifier|Window 1| Window 2| Window 3|
|-|-|-|-|
|`GaussianNB`| 79.65| 80.12| 81.60|
|`SVC` (`rbf`)|98.61|98.82|98.72|
|`LinearSVC`|97.90|98.21|98.11|

Even though the best scores were achieved by the SVC classifier with 'rbf' kernel, I have decided to use LinearSVC, because it is much faster and smaller (the size of the pickle with SVC(rbf) is over 40MB, LinearSVC is only 70KB, and LinearSVC pipeline is 6 times faster on my laptop then SVC(rbf)).

I augmented the training data by adding the flipped versions of images. I use `StratifiedShuffleSplit` to ensure even ratio of vehicles and non-vehicles data in the training set.

I fit three classifiers with three different sliding window sizes:

|Name|Pixels Per Cell|Cells Per Window| Total Window Size|
|-|-|-|-|
|Classifier 1| (8, 8)|(3, 3)| 24x24|
|Classifier 2| (8, 8)|(4, 4)| 32x32|
|Classifier 3| (8, 8)|(5, 5)| 40x40|

Original training images have shape (64, 64, 3), so I have to rescale it before I fit  classifiers. I also fit three scalers for the corresponding classifiers using `StandardScaler` from `sklearn` library.

Code contained in `clf_scaler.py`:
- Opens all training data,
- Extracts HOG feature vectors,
- Extracts Histogram feature vectors,
- Combine both feature vectors into one,
- Fits the scaler with a list of those vectors and transforms them,
- Fits the classifier with scaled vectors,
- Saves all classifiers and scalers to `clf_scaler.p` pickle file.

## Define the region of interest.

To save the computational resources I limit the region of interest to a region bounded by the box:

```python
roi = ((675, 400), (1280, 680))
```

Test image with marked region of interest:

![test_image][test_bounded]

And the ROI:

![ROI][test_roi]

## Implement sliding window technique.

I use a list of `steps` with steps in `x` and `y` direction for all fitted classifiers. Smaller steps improve the quality of the heat map, but also slow down the whole program:

I slide the smallest classifier window through the top of the ROI, medium classifier window through the middle and the largest window through the bottom of the ROI.

![sliding windows][sliding]


From every window I extract the hog features from the calculated hog map, extract the histogram features and feed it into the classifier fitted with vectors of appropriate size:

    window 0, feature vector shape=(352,)
    window 1, feature vector shape=(672,)
    window 2, feature vector shape=(1120,)

Every time classifier predicts the vector to belong to **vehicles** it adds the window to the list of the last 6 frames and uses all those positive windows to update the heat map.

## Draw rectangles around around detected vehicles.

By now the Heat Map should look like that:

![heatmap before][heat_pre]

To filter out some of the noise I use thresholding. Also I remove all bounding boxes with areas smaller than a threshold and ignore the false positives.

Now the Heat Map look more like that:

![heatmap after][heat_post]

Filtered Heat Map is used as a parameter of the `label` function from `scipy` library. That function splits the heat map into separate regions which look like:

![labels][label_map]

If everything goes right the output image looks like:
![boxes][label_boxes]

## Discussion.
This implementation is very fast and could not be applied in a real world. Using SVC with rbf kernel would improve the result but it would also render it unusable. This pipeline processes 6 frames of the video clip every second. Training data more suitable to the testing conditions would also improve the classifier scores. More training data, using DNNs to find the bounding boxes and faster implementation would be enough to make the pipeline usable in the real world.
