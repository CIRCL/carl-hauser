# Carl-Hauser Project

Open Source Testing Framework for image correlation, distance and analysis.
Strongly related to : [Douglas-Quaid](https://github.com/CIRCL/douglas-quaid)

# Problem statement (@CIRCL)

<img align="left" width="300" src="SOTA/carlhauser-doc/CarlHauser_cropped.svg">
A lot of information collected or processed by CIRCL are related to images (at large from photos, screenshots of website or screenshots of sandboxes). The datasets become larger and analysts need to classify, search and correlate throught all the images. 



## Target

Building a generic library and services which can establish correlation between pictures.
In order to achieve this goals, experiments needs to be conducted. This is the goal of this repository.

## Getting Started

* Review of existing algorithms, techniques and libraries for calculating distances between images, State Of The Art : [MarkDown](./SOTA/SOTA.md) | [PDF version](./SOTA/SOTA.pdf)

### Questions


<!-- 
- **_Can the library say there is no match or will it always gives a "top N" matching pictures ?_**

For our automated usecases, e.g. MISP, we need clear "answer" from the library, to allow automation. 
The final goal of this library is to map all matches into one of the three categories : Accepted pictures, To-review pictures, and Rejected pictures.
Therefore, the goal will be to reduce the "to review" category to diminish needed human labor to edgy cases only.

- **_What is this library about ? Similarity search (global picture to picture matching) ? Object search (object detection in scene, followed by object search among other pictures) ? ..._**

For a first iteration, we are focusing on picture-to-picture matching. Therefore, given a bunch of already known pictures, we want to know which pictures - if any - are close to a request picture. This is similar to an inverted-search engine.

The library is mainly intended to work with screenshots like pictures. However, it may be tweaked to provide similar services in a different context.
However, matching principles are quite similar, and the extension may be trivial.

- **_Can I use the library in the current state for production ?_**

Yes. However, be aware that the current state of the library is "beta". Therefore, stability and performances may not be as high as you would expect for an industry-standard image-matching tool.
You can import the client API in your own Python software and perfom API calls to push picture/request search/pull results.

See [Client Example](https://github.com/CIRCL/douglas-quaid/blob/master/carlhauser_client/core.py) if you want to start.



- **_Do we want to a "YES they are the same"/"NO they're not" algorithms output, which can deliver an empty set of results (threeshold at some point) OR  do we want a "top N" algorithm, who's trying to match the best pictures he has ? (ranking algorithm)_**

Depends on the usecase. MISP would need a certain clear correlation for automation. The "best match" output is mainly useful for quality evaluation of different algorithms. However some application could use it as a production output.

The final goal of this library is to map all matches into one of the three categories : Accepted pictures, To-review pictures, and Rejected pictures.
Therefore, the goal will be to reduce the "to review" category to diminish needed human labor to edgy cases only.

- **_Is it about a similarity search (global picture matching) or an object search (1 object -> Where is it within a scene OR one Scene -> Many objects -> Where each object is within other Scene ?)_**

For a first iteration, we are focusing on picture-to-picture matching. Given problems we will face and usecases we will add, the project may be extended to object to picture matching.
However, matching principles are quite similar, and the extension may be trivial.

- **_Can I use the library in the current state for production ?_**

Not now. The library has for now no "core element" that you can atomically use in your own software. The library is for now mainly a testbench to evaluate algorithms on your own dataset.
-->


### Prerequisites

See requirements.txt

(...)

### Installing

(...)

## Running the tests

(...)

## Running the benchmark evaluation

in /lib_testing you just have to launch "python3 ./launcher.py"
Parameters are hardcoded in the launcher.py, as : 
- Path to pictures folder
- Output folder to store results
- Requested outputs (result graphe, statistics, LaTeX export, threshold evaluation, similarity matrix ...)

This is currently working on most configuration and will explore following algorithms for matching : 
- ImageHash Algorithms (A-hash, P-hash, D-hash, W-hash ... )
- TLSH (be sure to have BMP pictures or uncompressed format at least. A function is available to convert pictures in /utility/manual.py) 
- ORB (and its parameters space)
- ORB Bag-Of-Words / Bag-Of-Features (and its parameters space, including size of the "Bag"/Dictionnary)
- ORB RANSAC (with/without homography matrix filtering)

You can also manually generate modified datasets from your original dataset : 
- Text detector and hider (DeepLearning, Tesseract, ...)
- Edge detector (DeepLearning, Canny, ...)
- PNG/BMP versions of pictures (compressed/uncompressed)

### For Developers

(...)

## Deployment

(...)

For the algorithms test library : See [installation instruction](./installation_info.md)

## Built With & Sources

* [Original project structure source](http://www.kennethreitz.org/essays/repository-structure-and-python)
* [Clean library implementation of algorithms](https://github.com/CIRCL/douglas-quaid)
* [Followed practice for logging](https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/)
* [Text detector model source](https://github.com/argman/EAST)
* [Built-for-the-occasion manual image classificator](https://github.com/Vincent-CIRCL/visjs_classificator)
* [Bibliography](https://www.zotero.org/groups/2296751/carl-hauser/items)

## Contributing
PR are welcomed.
New issues are welcomed if you have ideas or usecase that could improve the project.
