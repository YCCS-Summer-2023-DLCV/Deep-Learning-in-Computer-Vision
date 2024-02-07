# Deep Learning in Computer Vision
### Description
Advances in deep learning have revolutionized the field of computer vision and are driving innovations in many fast-growing industries, notably driverless cars, autonomous robots, and AR, to name a few. In this project, we will begin with a quick introduction to convolutional neural networks and learn to work with Tensorflow, Google’s framework for developing and training ML models. Together we selected a common problem in computer vision, namely image classification and segmentation.

### The Team
Hey! Our team was made up of Yeshiva University students taking this special summer course in Industrial Software Development. Please make sure to check out our LinkedIns: [Tuvya Macklin](https://www.linkedin.com/in/tuvyamacklin/), [Zachary Mankowitz](https://www.linkedin.com/in/zachary-mankowitz-a4a11324a/), [Eitan Traurig](https://www.linkedin.com/in/eitan-traurig-3332b2261/), [Jason Zelmanovich](https://www.linkedin.com/in/jasonzel/).

We couldn't have done this project without the generous help of our mentor [Gershom Kutliroff](https://www.linkedin.com/in/gershom-kutliroff-9a89611/). Gershom has been working in AI for over 15 years and was truley an invaluable resorce.

### Video Presentation
At the end of the course, we made a presentation with explanations of how and what we did to produce our results. Please feel free to see [our presentation](https://www.youtube.com/watch?v=JzgG1tagBnc) and [other groups](https://sacknovi.github.io/summer-2023/projects/teams.html#AI-1) who made projects too.

## Instance Segmentation and Image Classification Project
### Data
Our dataset was created based on the [COCO](https://cocodataset.org/#home) dataset. This dataset contains thousands of images with bounding boxes around the objects and labels for each object classifying what it is. We cropped the images using bounding boxes to generate images with only one type of object in them. We chose to only work on 10 classes (foods specifically) and ignored all other types of objects. Each crop was saved in a directory corresponding to it's class.

Our pipeline uses selective search on the images it receives to generate bounding boxes which are then evaluated by the model. Selective search does not provide high quality bounding boxes like COCO has. Therefore, the examples we generated did not train the model well for the crops it was actually used for. To address this issue, we generated a second version of the dataset using selective search to generate the bounding boxes. We took a few of the selctive search boxes that roughly matched the COCO boxes and used those to generate the cropped examples.

Most of our work was done using AWS EC2 instances. Therefore, it was natural for us to use AWS S3 buckets to share the datasets. This is what we did.

### Image Classification
Image classification consists of feeding an image to a trained model and then recieving an output representing the class prediction and the confidence of that prediction. Training took place on AWS EC2 instances specifically utilizing Nvidia GPUs for faster training.

We implementing this classification model using a pre-trained EfficentNet B2 model with fine tuning to increase accuracy for our specific class of objects (Transfer Learning). To address the issue of a small dataset and possible overfitting we utilized data augmentation and a dropout layer to increase the size of our dataset while also helping the model to generalize on input that was not as good as the training data. Our dataset was also imbalanced for some of the classes, to address this issue the loss function was changed from categorical cross entropy to focal loss which helps to assign more weight to the underepresented classes.

### Image Segmentation
Image segmentation is the process of labeling which object each pixel in an image belongs to; i.e. it highlights/outlines an object in an image. We began to implement this into our model. Due to time constraints we only had time to apply it to broccoli.

Image segmentation works using a U-Net architecture. This means that we first compress the image to an abstract representation (using a similar process to image classification), and then decompress it into the actual mask. As the image is decompressed, the model receives information about the original image. This allows it to make decisions about the mask based on the concrete appearance of the object in the image.
