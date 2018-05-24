|File|Description|
|-|-|
|clf_scaler.py| Fits classifers and scalers. Saves a pickle with them to `clf_scaler.p`|
|vehdet.py| Processes the frames. Uses fitted `clf_scaler.p`|
|clf_scaler.p| Pickle with a list of {classifer, scaler} pairs|


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
|Orientations|9|
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

I use Linear Support Vector Machine Classifier `LinearSVC` from the `sklearn` library. I use default parameters (`C=1.0`). I train two classifiers for two different sliding window sizes:

|Name|Pixels Per Cell|Cells Per Window| Total Window Size|
|-|-|-|-|
|Classifier 1| (8, 8)|(8, 8)| 64x64|
|Classifier 2| (8, 8)|(12, 12)| 96x96|

Original training images have shape (64, 64, 3), so I have to rescale it before I train the 2nd classifier. I also fit two scalers for the corresponding classifiers using `StandardScaler` from `sklearn` library.

Code contained in `clf_scaler.py`:
- Opens all training data,
- Extracts HOG feature vectors,
- Extracts Histogram feature vectors,
- Combine both feature vectors into one,
- Fits the scaler with a list of those vectors and transforms them,
- Fits the classifier with scaled vectors,
- Saves all classifiers and scalers to `clf_scaler.p` pickle file.

Scores of the fitted classifiers:

|Name|Score|
|-|-|
|Classifier 1| 0.9803|
|Classifier 2| 0.9648|

(NOTE: I fitted the classifiers and scalers on the Floydhub GPU instance. The sklearn library they use has the number 0.19.0, while `carnd-term1` environment use 0.19.1.)

## Define the region of interest.

To save the computational resources I limit the region of interest to a region bounded by the box:

```python
roi = ((600, 370), (1280, 680))
```

Test image with marked region of interest:

![test_image][test_bounded]

And the ROI:

![ROI][test_roi]

## Implement sliding window technique.

I use a list of `steps` with steps in `x` and `y` direction for the both fitted classifiers. Smaller steps improve the quality of the heat map, but also slows down the whole program:

```python
# window sizes used (pixels per cell, number of cells)
windows = [(8, 8), (8, 12)]

# steps in x and y direction for all used classifiers
steps = [(4, 6), (2, 2)]

# cv notation
roi = ((600, 370), (1280, 680))

span_x = roi[1][0] - roi[0][0] # span of roi along x axis
span_y = roi[1][1] - roi[0][1] # span of roi along y axis

ncells_x, ncells_y = [], [] # number of cells in roi for every window size
nwin_x, nwin_y = [], [] # number of windows in roi for every window size

for i, win in enumerate(windows):
    ncells_x.append(span_x // win[0])
    ncells_y.append(span_y // win[0])
    nwin_x.append((ncells_x[i] - win[1])//steps[i][0])
    nwin_y.append((ncells_y[i] - win[1])//steps[i][1])
```

I slide window over the region of interest, extract the hog features from the calculated hog map, extract the histogram features and feed it into the classifier fitted with vectors of appropriate size:

    window 0, feature vector shape=(3328,)
    window 1, feature vector shape=(7936,)

Every time classifier predicts the vector to belong to **vehicles** it updates the Heat Map by adding 1 to all points bounded by the *positive* window.

## Draw rectangles around around detected vehicles.

By now the Heat Map should look like that:

![heatmap before][heat_pre]

To filter out some of the noise I use thresholding. I scale all the values of the Heap Map matrix to be in the range 0 to 9. I use a trick to move values lower than the mean value closer to 0 and values greater than the mean value closer to 9.

```python
hmap = (((heat_map/np.max(heat_map)+0.5)**2)*4).astype(np.uint8)
hmap[hmap < 5] = 0
```

Now the Heat Map look more like that:

![heatmap after][heat_post]

Filtered Heat Map is used as a parameter of the `label` function from `scipy` library. That function splits the heatmap into separate regions which look like:

![labels][label_map]

Now, we can use a modified version of Udacity's short script to find the bounding box. Modifications remove all boxes with area smaller than a set value. Also, If the box is not in certain vicinity of any of the boxes from the previous frame, it is ignored and added to the *candidates* list for the following frame.

```python
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
```

If everything goes right the output image looks like:
![boxes][label_boxes]

## Discussion.
This implementation is slow and could not be applied in a real world. Training data was not good enough to fit the SVM classier even though it had high score on the test data. There are some regions on the roads that the classier with high confidence and persistence recognize as a vehicle. To fix this problem, more training data is needed for both vehicles and non-vehicles, although that could introduce another problem which is SVM's inability to online training. All data has to be fitted into computer's memory. To train classifier for 128x128 pixels,  classifier needed 40GB of RAM, and 96x96 pixels needed 12GB of RAM only to be fitted with the original training data.

Sometimes, the output is as expected:

![output 1][o1]

Or,

![output 2][o2]

But very often, it is very confident that a piece of the road is a car. It is not possible to filter that kind of false positive easily:

![output 3][o3]
