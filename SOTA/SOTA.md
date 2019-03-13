-   [Introduction](#introduction)
    -   [Problem Statement](#problem-statement)
-   [Classic Computer vision techniques - White box algorithms](#classic-computer-vision-techniques---white-box-algorithms)
    -   [Structure of Classic vision techniques](#structure-of-classic-vision-techniques)
    -   [Step 1 - Key Point Detection](#step-1---key-point-detection)
    -   [Step 2 - Descriptor Extraction](#step-2---descriptor-extraction)
    -   [Step 3 - Feature representation](#feature-representation)
    -   [Step 4 - Matching](#step-3---matching)
    -   [Step 5 - Model Fitting](#step-4---model-fitting)
-   [Global Features Algorithms](#standard-algorithms)
    -   [Full region features - FH or CTPH - Fuzzy Hashing Algorithms](#full-region-features---fh-or-ctph---fuzzy-hashing-algorithms)
    -   [Per subregion features](#per-subregion-features)
-   [Algorithms combination](#algorithms-combination)
    -   [Block-based approach + KeyPoint approach for Image manipulation](#block-based-approach-keypoint-approach-for-image-manipulation)
-   [Local Features Algorithms](#standard-algorithms)
    -   [Comparison overview](#comparison-overview)
    -   [Non-binary features](#non-binary-features)
    -   [Binary features](#binary-features)
    -   [Unsorted](#unsorted)
-   [Neural networks – Black box algorithms](#neural-networks-black-box-algorithms)
    -   [FAST – Features from Accelerated Segment Test](#fast-features-from-accelerated-segment-test)
    -   [CNN - Convolutional Neural Network](#cnn---convolutional-neural-network)
    -   [FRCNN - Faster RCNN](#frcnn)
    -   [RTSVMs - Robust Transductive Support Vector Machines](#rtsvms---robust-transductive-support-vector-machines)
    -   [RBM - Restricted Boltzmann machine](#rbm)
    -   [RPA - Robust Projection Algorith](#rpa)
    -   [Boosting SSC](#bssc)
    -   [ConvNet - Convolutional Neural Networks](#convnet---convolutional-neural-networks)
-   [Utility algorithms](#utilities-algorithms)
    -   [SWS - Sliding Windows Search](#sws)
    -   [ESS - Efficient Subwindow Search](#ess)
    -   [SLICO - Simple Linear Iterative Clustering](#slico)
    -   [HSNW - ... indexing](#hsnw---...-indexing)

Introduction
============

A general overview was made through standard web lookup. \[6\] A look was given to libraries, which also provide detailed and useful information. \[44\]

In the following, we expose :

-   The main steps of a Image Matching algorithm

-   Few of the most popular Image Matching algorithms

Please, be sure to consider this document is under construction, and it can contain mistakes, structural errors, missing areas .. feel free to ping me if you find such flaw. (Open a PR/Issue/...)

##### Problem Statement

\[7\] states the Image Retrieval problem as “Given a query image, finding and representing (in an ordered manner) the images depicting the same scene or objects in large unordered image collections”

Classic Computer vision techniques - White box algorithms
=========================================================

Correspondances found between images can be used for \[4\]:

1.  **Similarity Measurement** : probability of two images for showing the same scene

2.  **Geometry Estimation** : estimate the transformation between two object views

3.  **Data Association** : Sorting pictures by scene (TO CHECK : Same as Similarity measurement

Block based approach : the image is divided into various blocks. These block division is done on the basis of Discrete Wavelet Transform, Discrete Cosine Transform, Zernike moment, and Fourier Mellin Transform. \[25\]

![Image maching pipeline from \[4\] <span data-label="fig:image_matching_pipeline"></span>](sota-ressources/image-matching-pipeline.png)

#### Structure of Classic vision techniques

From \[33\] :

1.  **Global features detection**

    1.  **Full region**

    2.  **Per subregion**

2.  **Local features detection**
    Detection should be stable and repeatable. Corners, textured areas, etc. can be interest points. Robust to occlusion and viewpoint changes.

    1.  **Dense sampling over regular grid**

    2.  **Interest points detection**
        Find where interest points are.

        1.  Not robust to scale
            Examples : Harris corner detector

        2.  Robust to scale
            Examples : Not robust + Increasing Gaussian blur many time, one for each scale ; Automatic scale selection ; ...

    3.  **Exotics**

        1.  Random points sampling

        2.  Segmentation (?)

        3.  Pose estimation
            Example : “pictorial structure” (poselet) More complex

Step 1 - Key Point Detection
----------------------------

-   Corner detectors to find easily localizable points.

#### Harris Detector

From the original paper \[14\]. Based on the central principle: at a corner, the image intensity will change largely in multiple directions, with a windows shift.

(invariance to rotation, scale, illumination, noise .. said \[33\])

Distinctive features :

-   Rotation-invariant

-   NOT scaling invariant
    One point could be a corner in a small scaled neighborhood, or as an edge in a large scaled neighborhood.

#### FAST - Features from Accelerated Segment Test

From the original paper \[26\] cited in \[37\]
Is a corner detector, based on machine learning.

Distinctive features :

-   Not rotation-invariant (no orientation calculation)

-   ? scaling invariant

#### Pro

-   High repeatability

#### Con

-   not robust to high levels noise

-   can respond to 1 pixel wide lines

-   dependent on a threshold

Step 2 - Descriptor Extraction
------------------------------

Extract a small patch around the keypoints, preserving the most relevant information and discaring necessary information (illumination ..)

Can be :

-   Pixels values

-   Based on histogram of gradient

-   Learnt

Usually :

-   Normalized

-   Indexed in a searchable data structure

##### Example

Vector descriptors based on our keypoints, each descriptor has size 64 and we have 32 such, so our feature vector is 2048 dimension.

##### Descriptor’s quality

A good descriptor code would be, according to \[45\] :

-   easily computed for a novel input

-   requires a small number of bits to code the full dataset

-   maps similar items to similar binary codewords

-   require that each bit has a 50

We should be aware that a smaller code leads to more collision in the hash.

Step 3 - Feature representation
-------------------------------

A local feature needs to be represented. From \[33\]

### Bag-Of-Words or Bag-Of-Features

From \[33\], representing an image as a set of feature descriptor.

##### Pro

-   Insensitivity of objects location in image

##### Con

-   Loss of spatial information

### Codebook Generation

From \[33\], K-Means clustering over all words describing all pictures. A representative word (=Codeword) of each cluster is chosen (the “mean word”). A list of all representative words is created. A representative vector for each image, is created, as a boolean\_list/histogram of representative words linked or not to this image.

##### Pro

-   Shorten the comparisons to do (TO CHECK)

##### Con

-   Representation ambiguity : Codeword may not be representative of a cluster of words (too large, too narrow, more than 1 meaning, ...)

### Soft Vector Quantization

From \[33\], codebook Generation with most and least frequent words removal. Each feature is then represented by a small group of codewords.

##### Pro

-   Mitigate the representation ambiguity problem of CodeBook

##### Con

-   Undo something that has been done ? TO CHECK !

### Hierarchical CodeWords

From \[33\], keep spatial information about the neighboorhood of a codeword.

### Visual sentences

Project codeword on a spatial axis. Relative spatial relation between words are kept.

### SPM - Spatial Pyramid Matching

From \[33\], divide a picture into equal partitions (/4, /8, ..), compute a Bag-Of-Word for each partition, its histogram, and concatenate them into one big “ordered” histogram.

##### Pro

-   Keep spatial information of features

##### Con

-   Some “bad translation” can occurs, and split features into different Bag-of-words.

![3 levels spatial pyramid from \[33\] <span data-label="fig:spm_figure"></span>](sota-ressources/spm.png)

### L-SPM - Latent Spatial Pyramid Matching

From \[33\], based on SPM but does not split the picture in equal partition = the cell of the pyramid is not spatially fixed.

The cells of the pyramid to move within search regions instead of a predefined rigid partition. Use ESS (See utilities)

##### Pro

##### Con

-   High computaitonal cost

Step 4 - Matching
-----------------

Linked to correspondence problem ?

### Distance

#### Best match

-   Returns only the best match

-   Returns the K (parameter) best matchs

#### Hamming distance / Bruteforce

Partially solved by \[20\]
Works with binary features. Can be accelerated with GPU \[4\].

-   *O*(*N*<sup>2</sup>), N being the number of descriptor per image

-   One descriptor of the first picture is compared to all descriptor of a second candidate picture. A distance is needed. The closest is the match.

-   Ratio test

-   CrossCheck test : list of “perfect match” (TO CHECK)

#### FLANN – Fast Library for Approximate Nearest Neighboors

From \[4\], is an approximation for matching in Euclidian space, with KD-Tree techniques.
Work with non-binary features.

-   Collections of algorithm, optimized for large dataset/high dimension

-   Returns the K (parameter) best matchs

-   \[46\]

##### Implementation

Apache 2 From \[5\], available at : <https://github.com/nmslib/nmslib> \[42\] Does not benchmark memory usage.

### QBH - Quantization Based Hashing

From \[29\] Incorporates quantization error into the conventional property preserving hashing models to improve the effectiveness of the hash codes

#### IMI - Inverted Multi-Index

#### NGS - Neighboorhood Graph Search

#### HNSW - Hierarchical Navigable Small Worlds

Graph based approach. Precise approximate nearest neighbor search in billion-sized datasets.
**Highly scalable** : “Indexing 1 billion vectors takes about 26 hours with L&C: we can add more than 10,000 vectors per second to the index. We refine at most 10 vectors.”

##### Implementation

BSD <https://github.com/facebookresearch/faiss> \[43\]

### Selection

Higly noisy correspondences need to be filtered.

#### RATIO - 

From \[4\] recognizes the distinctiveness of features by comparing the distance of their two nearest neighbors.

#### GMS - Gird-based Motion Statistics

Uses the motion smoothness constraint. Equivalent to RATIO.
Robustness, accuracy, sufficiency, and efficiency of GMS all depend on the number of interest points detected.

### Compression of descriptors before matching

#### LSH – Locally Sensitive Hashing

-   O(~N)

-   Returns the K (parameter) best matchs

-   \[46\]

-   Convert descriptor (floats) to binary strings. Binary strings matched with Hamming Distance, equivalent to a XOR and bit count (very fast with SSE instructions on CPU)

#### BBF – Best bin first Kd-tree

-   O(~N)

-   Example : SIFT – Scale Invariant Feature Tranform

Step 5 - Model Fitting
----------------------

From \[4\] and \[44\], is a step where the geometry of the scene is verified and/or estimated. Given correspondances, the pose of the object is estimated.

-   Identify inliers and outliers ~ Fitting a homography matrix ~ Find the transformation of (picture one) to (picture two)

-   Inliers : “good” points matching that can help to find the transformation

-   outliers : “bad” points matching

### RANSAC – Random Sample Consensus

Estimation of the homography, searches for the best relative pose between images iteratively and removes outliers that disobey the estimated geometry relation finally. Correspondences that pass the geometry verification are named verified correspondences. Provides a robust estimation of transform matrix.

### Least Meadian

Global Features Algorithms
==========================

Generally, weak against occlusion, clutter. Need fixed viewpoint, clear background,fixed pose.

The main assumption is that the similar images in the Euclidean space must have similar binary codes. \[7\]

Categories :

-   Locality Sensitive Hashing schemes (LSH)

-   Context Triggered Piecewise Hashing (CTPH)

Full region features - FH or CTPH - Fuzzy Hashing Algorithms
------------------------------------------------------------

The following algorithms does not intend to match pictures with common part, but to match pictures which are rougly the same. To be clear : If the hashes are different, then the data is different. And if the hashes are the same, then the data is likely the same. There is a possibility of a hash collision, having the same hash values then does not guarantee the same data.

Discrete Cosine Transformation (CDT) may be worst than Discrete Wavelet Transformation (DWT).

From \[28\], also called Context Triggered Piecewise Hashing (CTPH). It is a combination of Cryptographic Hashes (CH), Rolling Hashes (RH) and Piecewise Hashes (PH).

Fuzzy hashing has as a goal of identifying two files that may be near copies of one another.

SDHash seesm the more accurate, but the slower. Cross-reference seems a good way to go.

Examples : Holistic features (=“Spatial enveloppe” = naturalness, openness, roughness, ruggedness, expansion ..), colors histograms, “Global Self-Similarity” (=spatial arrangement)

### A-HASH : Average Hash

From ... \[34\] : “the result is better than it has any right to be.”

relationship between parts of the hash and areas of the input image = ability to apply “masks” (like “ignore the bottom 25% of the image”.) and “transformations” at comparison time. (searches for rotations in 90degree steps, mirroring, inverts...)

8 bits for a image vector.

Idea to be faster (achieve membw-bound conditions) : Batch search (compare more than one vector to all others) = do X search at the same time.

More than one vector could be transformation of the initial image (rotations, mirrors).

##### Pro

-   Masks and transformation available

-   Ability to look for modified version of the initial picture

-   Only 8 bits for a image vector.

##### Implementation

ImageHash 4.0 <https://pypi.org/project/ImageHash/>

##### Implementation

Javascript Implementation : \[1\]

##### Results

![Top to bottom : structural matching leading to missmatch, structural matching leading to match, structural matching seeing “a form”, missing logo in the match<span data-label="fig:tests"></span>](sota-ressources/outputs-evaluation/a-hash/false_structural_2.png "fig:") ![Top to bottom : structural matching leading to missmatch, structural matching leading to match, structural matching seeing “a form”, missing logo in the match<span data-label="fig:tests"></span>](sota-ressources/outputs-evaluation/a-hash/good_structural.png "fig:") ![Top to bottom : structural matching leading to missmatch, structural matching leading to match, structural matching seeing “a form”, missing logo in the match<span data-label="fig:tests"></span>](sota-ressources/outputs-evaluation/a-hash/false_structural.png "fig:") ![Top to bottom : structural matching leading to missmatch, structural matching leading to match, structural matching seeing “a form”, missing logo in the match<span data-label="fig:tests"></span>](sota-ressources/outputs-evaluation/a-hash/no_logo_match.png "fig:")

###### Time

Hashing time : 16.796968936920166 sec for 207 items (0.081s per item)
Matching time : nobs=207, minmax=(0.023s, 1.58s), mean=0.08s, variance=0.025s, skewness=6.62s, kurtosis=50.37s

### D-HASH - Difference Hashing

From \[13\], DHash is a very basic algorithm to find nearly duplicate pictures.
The hash can be of length 128 or 512 bits. The delta between 2 “matches” is a Hamming distance (\# of different bits.)

##### Pro

-   Detecting near or exact duplicates : slightly altered lighting, a few pixels of cropping, or very light photoshopping

##### Con

-   Not for similar images

-   Not for duplicate-but-cropped

##### Steps of the algorithm

1.  Convert the image to grayscale

2.  Downsize it to a 9x9 thumbnail

3.  Produce a 64-bit “row hash”: a 1 bit means the pixel intensity is increasing in the x direction, 0 means it’s decreasing

4.  Do the same to produce a 64-bit “column hash” in the y direction

5.  Combine the two values to produce the final 128-bit hash value

##### Implementation

ImageHash 4.0 <https://pypi.org/project/ImageHash/>

### P-HASH - Perceptual Hash

From ... \[34\] and \[15\] and \[36\]
Exist in mean and median flavors
8 bits for a image vector.
Java implementation : \[35\]

##### Pro

-   Robustness to gamma

-   Robustness to color histogram adjustments

-   Should be robust to rotation, skew, contrast adjustment and different compression/formats.

-   , C++, API

##### Steps of the algorithm

1.  Reduce size of the input image to 32x32 (needed to simplify DCT computation)

2.  Reduce color to grayscale (same)

3.  Compute the DCT : convert image in frequencies, a bit similar to JPEG compression

4.  Reduce the DCT : keep the top-left 8x8 of the DCT, which are the lowest frequencies

5.  Compute the average DCT value : without the first term (i.e. solid colors)

6.  Further reduce the DCT : Set the 64 hash bits to 0 or 1 depending on whether each of the 64 DCT values is above or below the average value.

7.  Construct the hash : create a 64 bits integer from the hash

8.  Comparing with Hamming Distance (threeshold = 21)

##### Implementation

ImageHash 4.0 <https://pypi.org/project/ImageHash/>

### W-HASH - Wavelet Hash

From ... Uses DWT instead of DCT. \[TO LOOK\]

##### Implementation

ImageHash 4.0 <https://pypi.org/project/ImageHash/>

### SimHash - Charikar’s simhash

From ... \[20\]
repository of 8B webpages, 64-bit simhash fingerprints and k = 3 are reasonable.
C++ Implementation

### R-HASH

From ... \[34\]

Equivalent to A-Hash with more granularity of masks and transformation. Ability to apply “masks” (color channel, ignoring (f.ex. the lowest two) bits of some/all values) and “transformations” at comparison time. (color channel swaps)

48 bits for a rgb image vector

##### Pro

-   Masks and transformation available

-   More precise masks (than A-hash)

-   More precise transformations (than A-hash)

##### Con

-   Larger memory footprint

##### Steps of the algorithm

1.  Image scaled to 4x4

2.  Compute vector

3.  Comparison = sum of absolute differences: abs(a\[0\]-b\[0\]) + abs(a\[1\]-b\[1\]) + ... + abs(a\[47\]-b\[47\]) = 48 dimensional manhattan distance

### Spectral-HASH

From \[45\]. A word is given in \[34\]

The bits are calculated by thresholding a subset of eigenvectors of the Laplacian of the similarity graph

Similar performance to RBM

![Spectral Hashing comparison from \[45\] <span data-label="fig:spectral_hashing_comparison"></span>](sota-ressources/spectral_hashing_comparison.png)

### LSH - Locality Sensitve Hashing

Same as E2LSH ? Chooses random projections so that two closest image samples in the feature space fall into the same bucket with a high probability, from \[7\]

### E2LSH - LSH - Locality Sensitve Hashing

From \[12\] a word is given in \[45\] and \[7\].
The code is calculated by a random linear projection followed by a random threshold, then the Hamming distance between codewords will asymptotically approach the Euclidean distance between the items.

Not so far from Machine Learning Approaches, but outperformed by them.

##### Pro

-   Faster than Kdtree

##### Con

-   Very inefficient codes (512 bits for a picture (TO CHECK))

### TLSH - Locality Sensitve Hashing

From \[24\]

##### Pro

-   Parametered threeshold (below 30 in original paper)

-   Open Source

### Nilsimsa hash - Locality sensitive hash

A word in \[24\]

##### Pro

-   Open Source

### SSDeep - Similarity Digest 

From ... few words on it in \[28\]

Implementation (C) at \[41\]

Historically the first fuzzing algorithm. CTPH type.

![ Hashing time from \[17\] <span data-label="fig:ssdeep_timing"></span>](sota-ressources/ssdeep_time.png)

##### Pro

-   Effective for text (Spam, ..)

-   Open Source

##### Con

-   Not effective for Images, Videos, ...

-   Les effective than Sdhash

##### Steps of the algorithm

1.  Rolling hashing to split document into “6 bits values segments”

2.  Uses hash function (MD5, SHA1, ..) to produce a hash of each segment

3.  Concatenate all hashes to create the signature (= the fuzzy hash)

### SDHash - Similarity Digest Hash

From ... Roussev in 2010 few words on it in \[28\]

Uses Bloom Filters to identify similarities between files on condition with common features. (Quite blurry)

##### Pro

-   More accurate than VHash, SSDeep, MRSHV2

-   Options available (TO CHECK) - See a particular implementation used in \[28\]

-   Open Source

##### Con

-   Slow compared to MVHash, SSDeep, MRSHV2

##### Steps of the algorithm

1.  Perform a hash/entropy (TO CHECK) calculation with a moving window of 64 bits.

2.  Features (? How are they recognized?) are hashed with SHA-1

3.  Features are inserted into a Bloom Filter

### MVHash - Majority Vote Hash

From ... few words on it in \[28\]

It is Similarity Preserving Digest (SPD) Uses Bloom Filters

##### Pro

-   almost as fast as SHA-1 (paper)

##### Steps of the algorithm

1.  Majority vote on bit level (transformation to 0s or 1s)

2.  RLE = Run Length Encoding, represents 0s and 1s by their length

3.  Create the similarity digest (? TO CHECK)

### MRSH V2 - MultiResolution Similarity Hashing

From ... few words on it in \[28\] Variation of SSDeep, with polynomial hash instead of rolling hash (djb2)

##### Pro

-   Fast than SDHash

##### Con

-   Slow compared to MVHash, SSDeep

### GIST - 

From \[23\] a word in \[18\]

Holistic feature which is based on a low dimensional representation of the scene that does not require any form of segmentation, and it includes a set of perceptual dimensions (naturalness, openness, roughness, expansion, ruggedness)

Per subregion features
----------------------

**Per subregion** Example : Histogram of Oriented Gradients (HOG)

Holistic feature ...

### HOG - Histogram of Oriented Gradients

From ... A word in \[33\] The idea is to describe shape information by gradient orientation in localized sub-regions.

Algorithms combination
======================

Block-based approach + KeyPoint approach for Image manipulation
---------------------------------------------------------------

From \[25\]

Local Features Algorithms
=========================

Goal is to transform visual information into vector space

Comparison overview
-------------------

![Benchmarking of SIFT, SURF, ORB, AKAZE with RATIO and GMS selection ; FLANN or Hamming for distance. SP curves show the success ratio or success number (number of correspondance for AP) with thresholds. X-Axis being the threeshold. AP curves illustrate the mean number of verified correspondences with thresholds.\[4\] <span data-label="fig:benchmarking1"></span>](sota-ressources/benchmarking_1.png)

![Benchmarking of SIFT, SURF, ORB, AKAZE, BRISK, KAZE with computation time. Ordered best time from best to worst : red, green, blue, black. \[4\] <span data-label="fig:benchmarking2"></span>](sota-ressources/benchmarking_2.png)

![Benchmarking of SIFT, SURF, ORB, AKAZE, BRISK, KAZE on robustness (RS), accuracy (AS), sufficiency (SS). Ordered best time from best to worst : red, green, blue, black. \[4\] <span data-label="fig:benchmarking3"></span>](sota-ressources/benchmarking_3.png)

In few words :

-   **Robustness**
    Success ratio (15difference max from real position) = Succes to match pairs
    Non-binary are better than binaries algorithms. Number of interest points change the best matching method to choose.

-   **Accuracy**
    Success ratio (5difference max from real position) = Are pairs are matched “for sure”
    Non-binary are better than binaries algorithms

-   **Sufficiency**
    Mean number of correctly geometric matched pairs.
    ORB-GMS is the best.

-   **Efficiency**
    Feature detection time + Feature matching time.
    ORB and BRISK are the fastest, KASE the slowest.

Non-binary features
-------------------

### SIFT- Scale Invariant Feature Transform

From the original paper \[19\] and a concise explanation from \[38\] 3x less fast than Harris Detector

SIFT detects scale-invariant key points by finding local extrema in the difference-of-Gaussian (DoG) space. \[18\]
Each key point is described by a 128-dimensional gradient orientation histogram. Subsequently, all SIFT descriptors are modeled/quantized using a bag-of-words (BoW). The feature vector of each image is computed by counting the frequency of the generated visual words in the image.

Interesting “usability” notices are presented in \[31\], as skiping first octave features, ...

#### Con

-   and not included in OpenCV (only non-free module)

-   Slow (HOW MUCH TO CHECK)

#### Steps of the algorithm

1.  Extrema detection

    Uses an approximation of LoG (Laplacian of Gaussian), as a Difference of Gaussion, made from difference of Gaussian blurring of an image at different level of a Gaussian Pyramid of the image. Kept keypoints are local extrema in the 2D plan, as well as in the blurring-pyramid plan.

2.  Keypoint localization and filtering

    Two threesholds has to be set :

    -   Contract Threeshold : Eliminate low contract keypoint ( 0.03 in original paper)

    -   Edge Threeshold : Eliminate point with a curvature above the threeshold, that could match edge only. (10 in original paper)

3.  Orientation assignement

    Use an orientation histogram with 36 bins covering 360 degrees, filled with gradient magnitude of given directions. Size of the windows on which the gradient is calculated is linked to the scale at which it’s calculated. The average direction of the peaks above 80% of the highest peak is considered to calculate the orientation.

4.  Keypoint descriptors

    A 16x16 neighbourhood around the keypoint is divided into 16 sub-blocks of 4x4 size, each has a 8 bin orientation histogram. 128 bin are available for each Keypoint, represented in a vector of float, so 512 bytes per keypoint. Some tricks are applied versus illumination changes, rotation.

5.  Keypoint Matching

    Two distance calculation :

    -   Finding nearest neighboor.

    -   Ratio of closest distance to second closest is taken as a second indicator when second closest match is very near to the first. Has to be above 0.8 (original paper) (TO CHECK what IT MEANS)

### SIFT-FLOW

From \[11\], realize in motion prediction from a single image, motion synthesis via object transfer and face recognition.

##### Implementation

SIFT Flow (modified version of SIFT) C++ \[47\] at <http://people.csail.mit.edu/celiu/SIFTflow/>

### Root-SIFT

From \[2\] Better performances as SIFT, but no direct implementation found.

### SURF – Speeded-Up Robust Features

\[3\] Use the BoWs to generate local features.

#### Pro

-   Faster than SIFT (x3) : Parralelization, integral image ..

-   Tradeoffs can be made :

    -   Faster : no more rotation invariant, lower precision (dimension of vectors)

    -   More precision : **extended** precision (dimension of vectors)

-   Good for blurring, rotation

#### Con

-   -   Not good for illumination change, viewpoint change

#### Steps of the algorithm

1.  Extrema detection

    Approximates Laplacian of Guassian with a Box Filter. Computation can be made in parrallel at different scales at the same time, can use integral images … Roughly, does not use a gaussian approximation, but a true “square box” for edge detection, for example.

    The sign of the Laplacian (Trace of the Hessian) give the “direction” of the contrast : black to white or white to black. So a negative picture can match with the original ? (TO CHECK)

2.  Keypoint localization and filtering

3.  Orientation assignement

    Dominant orientation is computed with wavlet responses with a sliding window of 60

4.  Keypoint descriptors

    Neighbourhood of size 20sX20s is taken around the keypoint, divided in 4x4 subregions. Wavelet response of each subregion is computed and stored in a 64 dimensions vector (float, so 256 bytes), in total. This dimension can be lowered (less precise, less time) or extended (e.g. 128 bits ; more precise, more time)

5.  Keypoint Matching

### U-SURF – Upright-SURF

Rotation invariance can be “desactivated” for faster results, by bypassing the main orientation finding, and is robust up to 15rotation.

### GSSIS - Generalized Scale-Space Interest Points

From \[40\], generalized interest point, with colors exension, of SIFT and SURF.

Roughly : uses more complicated way of generating local interest points.

#### Pro

-   Scale-invariant

### LBP - Local Binary Pattern

From \[18\], use the BoWs to generate local features

Binary features
---------------

### ORB – Oriented FAST and Rotated BRIEF

From \[27\] which is rougly a fusion of FAST and BRIEF. See also \[39\]

#### Pro

-   Not patented

#### Steps of the algorithm

1.  Extrema detection

    FAST algorithm (no orientation)

2.  Keypoint localization and filtering

    Harris Corner measure : find top N keypoints

    Pyramid to produce multi scale features

3.  Orientation assignement

    The direction is extracted from the orientation of the (center of the patch) to the (intensity-weighted centroid fo the patch). The region/patch is circular to improve orientation invariance.

4.  Keypoint descriptors

    R-BRIEF is used, as Brief Algorithm is bad at rotation, on rotated patches of pixel, by rotating it accordingly with the previous orientation assignement.

5.  Keypoint Matching

    Multi-probe LSH (improved version of LSH)

### BRISK - 

### AKASE - 

Unsorted
--------

### SUSAN

From ... a word in \[26\]

### PSO

From ... few words in \[22\]

### SKF

From \[22\]

Faster than PSO.

### RPM - Robust Point Matching

From ... Few words in \[32\] Unidirectional matching approach. Does not “check back” if a matching is correct. Seems to achieve only the transformation (geometry matching) part.

### BRIEF – Binary Robust Independent Elementary Features

Extract binary strings equivalent to a descriptor without having to create a descriptor

See BRIEF \[48\]

#### Pro

-   Solve memory problem

#### Con

-   Only a keypoint descriptor method, not a keypoint finder

-   Bad for large in-plan rotation

#### Steps of the algorithm

1.  Extrema detection

2.  Keypoint localization and filtering

3.  Orientation assignement

4.  Keypoint descriptors

    Compare pairs of points of an image, to directly create a bitstring of size 128, 256 ou 512 bits. (16 to 64 bytes)

    Each bit-feature (bitstring) has a large variance ad a mean near 0.5 (TO VERIFY). The more variance it has, more distinctive it is, the better it is.

5.  Keypoint Matching Hamming distance can be used on bitstrings.

### R-BRIEF – Rotation (?) BRIEF

Variance and mean of a bit-feature (bitstring) is lost if the direction of keypoint is aligned (TO VERIFY : would this mean that there is a preferential direction in the pair of point selection ? )

Uncorrelated tests (TO CHECK WHAT IT IS) are selected to ensure a high variance.

### CenSurE

### KASE - 

Shipped in OpenCV library. Example can be found at \[21\]

#### Steps of the algorithm

1.  Extrema detection

2.  Keypoint localization and filtering

3.  Orientation assignement

4.  Keypoint descriptors

5.  Keypoint Matching

### Delaunay Graph Matching

Algorithm from 2012, quite advanced. Would need some tests or/and review See M1NN \[10\] that is presenting 3 algorithms :

- **M1NN Agglomerative Clustering**
Different types of data,robust to noise, may ’over’ cluster. Better clustering performance and is extendable to many applications, e.g. data mining, image segmentation and manifolding learning.

- **Modified Log-likelihood Clustering**
Measure and compare clusterings quantitatively and accurately. Energy of a graph to measure the complexity of a clustering.

- **Delaunay Graph Characterization and Graph-Based Image Matching**
Based on diffusion process and Delaunay graph characterization, with critical time. Graph-based image matching method. SIFT descriptors also used. Outperforms SIFT matching method by a lower error rate.

#### Pro

-   Lower error

-   Extensible to 3D (but not done yet ?)

#### Con

-   Lower number of matches

### Fast Spectral Ranking

From \[16\] Seems to have quite fast result, ranking algorithm. Still dark areas over the explanations.

### GHM - Generalized Hierarchical Matching Framework

From \[8\] Roughly, the algorithm split the input picture into interest areas, and then do matching on these different areas.

This tries to achieve a object-oriented recognition. It uses Saliency Map.

This (TO CHECK) is a non-rectangle version of SPM.

![Hierarchical Hashing as showed in \[8\] <span data-label="fig:generalized-matching"></span>](sota-ressources/hierarchical-matching.png)

#### Steps of the algorithm

1.  Multiple scale detection is performed in each image and the obtained multi-scale scores are averaged to get final single object confidence map.

Neural networks – Black box algorithms
======================================

See \[18\] to get a larger overview of deeplearning capabilities, applied to a particular field.

FAST – Features from Accelerated Segment Test
---------------------------------------------

From \[37\] the algorithm is mainly Machine Learning, but as far as I can see, there is no direct need of machine learning in the algorithm, but for speed.

It seems that the selection of candidate pixel, and the selection of a threeshold is holded by Machine Learning. It also seems, that “mostly brighter”, “similar” and “mostly darker” pixels are used to feed a decision tree (ID3 algorithm - decision tree classifier) to allow a fast recognition of a corner.

![Corner detector from \[26\] <span data-label="fig:spectral_hashing_comparison"></span>](sota-ressources/corner-detector.png)

#### Pro

-   “High performance” (HOW MUCH, TO CHECK)

#### Con

-   “Too” sensitive if n&lt;12 : increase in false-positive

-   Many calculation just to “throw away” a pixel.

-   Many True-postive around the same position

-   Not robust to high levels of noise

-   Dependant on a threshold

#### Steps of the algorithm

1.  Extrema detection For each pixel, select a cicle-patch (not disk-patch, not a surface!) of 16 pixels around it. The pixel is a corner if there is n (n=12) contiguous pixels parts of the circle, which are all brighter or all darker than the center-pixel.

    It’s easy to remove all “not-corner” points, by checking only few (1, 9, 5 and 13) pixels of the circle.

2.  Keypoint localization and filtering

3.  Orientation assignement

4.  Keypoint descriptors

5.  Keypoint Matching

CNN - Convolutional Neural Network
----------------------------------

From ... \[12\]

FRCNN - Faster RCNN
-------------------

From ... \[30\] Mainly for faces detection.

#### Pro

-   M

RTSVMs - Robust Transductive Support Vector Machines
----------------------------------------------------

From \[12\] Seems to scale very well (&gt;1 Million data)

Uses a hashing method, binary hierarchical trees and TSVM classifier.

![Biary hierarchical tree from \[12\] <span data-label="fig:spectral_hashing_comparison"></span>](sota-ressources/rtsvms.png)

RBM - Restricted Boltzmann machine
----------------------------------

From ... A word is given in \[45\]

To learn 32 bits, the middle layer of the autoencoder has 32 hidden units Neighborhood Components Analysis (NCA) objective function = refine the weights in the network to preserve the neighborhood structure of the input space.

#### Pro

-   More compact outputs code of picture than E2LSH = Better performances

RPA - Robust Projection Algorith
--------------------------------

From ... \[15\]

Boosting SSC
------------

From ... A word is given in \[45\]

#### Pro

-   Better than E2LSH

#### Con

-   Worst than RBM

ConvNet - Convolutional Neural Networks
---------------------------------------

Learn a metric between any given two images. The distance can be threesholded to decide if images match or not.

#### Training phase

Goal :

-   Minimizing distance between “same image” examples

-   Maximizing distance between “not same image” examples

#### Evaluation phase

Apply an automatic threeshold.

##### SVM - Support Vector Machine

Utility algorithms
==================

SWS - Sliding Windows Search
----------------------------

From ... \[33\] A bounding box is sliding on the picture, and an objet-existence score in the bounding box is computed for each position, and each rectangle size.

#### Pro

-   B

#### Con

-   Too complex ! *O*(*N*<sup>4</sup>) windows to evaluate, with N = resolution on one axis of the picture

Heuristics can be used to reduce the expected complexity of the algorithm. The picture is reduced in size, with a constant size bounding box, to find objects at different scales. These heuristics may miss objects.

ESS - Efficient Subwindow Search
--------------------------------

From \[33\] Based on a branch-and-bound algorithm. The algorithm does not evaluate all subrectangle of rectangle with a low evaluation of the best chance they have to contain an object.

#### Pro

-   Sublinear to number of pixels. ( below *O*(*N*) )

SLICO - Simple Linear Iterative Clustering
------------------------------------------

Cluster a picture into smaller chunks. For example, used in \[25\] for Copy Move detection.

HSNW - ... indexing
-------------------

From ... A word in \[9\]

1. Valentino Aluigi. 2019. JavaScript implementation of the Average Hash using HTML5 Canvas.

2. R. Arandjelović and A. Zisserman. 2012. Three things everyone should know to improve object retrieval. In *2012 IEEE Conference on Computer Vision and Pattern Recognition*, 2911–2918. <https://doi.org/10.1109/CVPR.2012.6248018>

3. Herbert Bay, Tinne Tuytelaars, and Luc Van Gool. 2006. SURF: Speeded Up Robust Features. In *Computer Vision 2006*, Aleš Leonardis, Horst Bischof and Axel Pinz (eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 404–417. <https://doi.org/10.1007/11744023_32>

4. JiaWang Bian, Le Zhang, Yun Liu, Wen-Yan Lin, Ming-Ming Cheng, and Ian D Reid. Image Matching: An Application-oriented Benchmark. 11.

5. Leonid Boytsov and Bilegsaikhan Naidan. 2013. Engineering Efficient and Effective Non-metric Space Library. In *Similarity Search and Applications*, David Hutchison, Takeo Kanade, Josef Kittler, Jon M. Kleinberg, Friedemann Mattern, John C. Mitchell, Moni Naor, Oscar Nierstrasz, C. Pandu Rangan, Bernhard Steffen, Madhu Sudan, Demetri Terzopoulos, Doug Tygar, Moshe Y. Vardi, Gerhard Weikum, Nieves Brisaboa, Oscar Pedreira and Pavel Zezula (eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 280–293. <https://doi.org/10.1007/978-3-642-41062-8_28>

6. Chomba Bupe. 2017. What algorithms can detect if two images/objects are similar or not? - Quora.

7. Hakan Cevikalp, Merve Elmas, and Savas Ozkan. 2018. Large-scale image retrieval using transductive support vector machines. *Computer Vision and Image Understanding* 173: 2–12. <https://doi.org/10.1016/j.cviu.2017.07.004>

8. Qiang Chen, Zheng Song, Yang Hua, Zhongyang Huang, and Shuicheng Yan. 2012. Hierarchical matching with side information for image classification.

9. Matthijs Douze, Alexandre Sablayrolles, and Herve Jegou. 2018. Link and Code: Fast Indexing with Graphs and Compact Regression Codes. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 3646–3654. <https://doi.org/10.1109/CVPR.2018.00384>

10. Yan Fang. 2012. Data Clustering and Graph-Based Image Matching Methods.

11. William T. Freeman, Antonio Torralba, Jenny Yuen, and Ce Liu. 2010. SIFT Flow: Dense Correspondence across Scenes and its Applications.

12. Aristides Gionis, Piotr Indyk, and Rajeev Motwani. Similarity Search in High Dimensions via Hashing. 12.

13. Nicolas Hahn. 2019. Differentiate images in python: Get a ratio or percentage difference, and generate a diff image - nicolashahn/diffimg.

14. C. Harris and M. Stephens. 1988. A Combined Corner and Edge Detector. In *Procedings of the Alvey Vision Conference 1988*, 23.1–23.6. <https://doi.org/10.5244/C.2.23>

15. Igor. 2011. Nuit Blanche: Are Perceptual Hashes an instance of Compressive Sensing ? *Nuit Blanche*.

16. Ahmet Iscen, Yannis Avrithis, Giorgos Tolias, Teddy Furon, and Ondrej Chum. 2018. Fast Spectral Ranking for Similarity Search. In *2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 7632–7641. <https://doi.org/10.1109/CVPR.2018.00796>

17. Jesse Kornblum. 2006. Identifying almost identical files using context triggered piecewise hashing. *Digital Investigation* 3: 91–97. <https://doi.org/10.1016/j.diin.2006.06.015>

18. Zhongyu Li, Xiaofan Zhang, Henning Müller, and Shaoting Zhang. 2018. Large-scale retrieval for medical image analytics: A comprehensive review. *Medical Image Analysis* 43: 66–84. <https://doi.org/10.1016/j.media.2017.09.007>

19. David G. Lowe. 2004. Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision* 60, 2: 91–110. <https://doi.org/10.1023/B:VISI.0000029664.99615.94>

20. Gurmeet Singh Manku, Arvind Jain, and Anish Das Sarma. 2007. Detecting near-duplicates for web crawling. In *Proceedings of the 16th international conference on World Wide Web - WWW ’07*, 141. <https://doi.org/10.1145/1242572.1242592>

21. Andrey Nikishaev. 2018. Feature extraction and similar image search with OpenCV for newbies. *Medium*.

22. Ann Nurnajmin Qasrina, Dwi Pebrianti, Ibrahim Zuwairie, Bayuaji Luhur, and Mat Jusof Mohd Falfazli. 2018. Image Template Matching Based on Simulated Kalman Filter (SKF) Algorithm.

23. Aude Oliva and Antonio Torralba. Modeling the Shape of the Scene: A Holistic Representation of the Spatial Envelope. 31.

24. Jonathan Oliver, Chun Cheng, and Yanggui Chen. 2013. TLSH – A Locality Sensitive Hash. In *2013 Fourth Cybercrime and Trustworthy Computing Workshop*, 7–13. <https://doi.org/10.1109/CTC.2013.9>

25. Reshma Raj and Niya Joseph. 2016. Keypoint Extraction Using SURF Algorithm for CMFD.

26. Edward Rosten and Tom Drummond. 2006. Machine Learning for High-Speed Corner Detection. In *Computer Vision 2006*, Aleš Leonardis, Horst Bischof and Axel Pinz (eds.). Springer Berlin Heidelberg, Berlin, Heidelberg, 430–443. <https://doi.org/10.1007/11744023_34>

27. Ethan Rublee, Vincent Rabaud, Kurt Konolige, and Gary Bradski. 2011. ORB: An efficient alternative to SIFT or SURF. In *2011 International Conference on Computer Vision*, 2564–2571. <https://doi.org/10.1109/ICCV.2011.6126544>

28. Nikolaos Sarantinos, Chafika Benzaid, Omar Arabiat, and Ameer Al-Nemrat. 2016. Forensic Malware Analysis: The Value of Fuzzy Hashing Algorithms in Identifying Similarities. In *2016 IEEE Trustcom/BigDataSE/ISPA*, 1782–1787. <https://doi.org/10.1109/TrustCom.2016.0274>

29. Jingkuan Song, Lianli Gao, Li Liu, Xiaofeng Zhu, and Nicu Sebe. 2018. Quantization-based hashing: A general framework for scalable image and video retrieval. *Pattern Recognition* 75: 175–187. <https://doi.org/10.1016/j.patcog.2017.03.021>

30. Xudong Sun, Pengcheng Wu, and Steven C.H. Hoi. 2018. Face detection using deep learning: An improved faster RCNN approach. *Neurocomputing* 299: 42–50. <https://doi.org/10.1016/j.neucom.2018.03.030>

31. Sahil Suri, Peter Schwind, Johannes Uhl, and Peter Reinartz. 2010. Modifications in the SIFT operator for effective SAR image matching.

32. Xuan Yang, Jihong Pei, and Jingli Shi. 2014. Inverse consistent non-rigid image registration based on robust point set matching.

33. Pengfei Yu. 2011. Image classification using latent spatial pyramid matching.

34. 2011. Looks Like It - The Hacker Factor Blog.

35. 2011. pHash-like image hash for java. *Pastebin.com*.

36. 2013. pHash.Org: Home of pHash, the open source perceptual hash library.

37. 2014. FAST Algorithm for Corner Detection 3.0.0-dev documentation.

38. 2014. Introduction to SIFT (Scale-Invariant Feature Transform) 3.0.0-dev documentation.

39. 2014. ORB (Oriented FAST and Rotated BRIEF) 3.0.0-dev documentation.

40. 2015. Image Matching Using Generalized Scale-Space Interest Points.

41. 2019. Fuzzy hashing API and fuzzy hashing tool. Contribute to ssdeep-project/ssdeep development by creating an account on GitHub.

42. 2019. Non-Metric Space Library (NMSLIB): An efficient similarity search library and a toolkit for evaluation of k-NN methods for generic non-metric spaces.: nmslib/nmslib.

43. 2019. A library for efficient similarity search and clustering of dense vectors.: facebookresearch/faiss.

44. Feature Matching + Homography to find Objects 3.0.0-dev documentation.

45. Spectralhashing.Pdf.

46. OpenCV: Feature Matching.

47. SIFT Flow: Dense Correspondence across Scenes and its Applications.

48. BRIEF (Binary Robust Independent Elementary Features) 3.0.0-dev documentation.
