# PREQA: A Photorealistic Dataset for Embodied Question Answering

PREQA is a high-quality dataset that is designed to bring the task of Embodied Question Answering (EQA) as close as possible to the real-world. As such, its
primary focus is to be used as a test set for pre-trained EQA systems. The dataset consists of images of a real-world robot lab and introduces new features such as camera tilt, viewpoint-based navigation and novel annotations. The dataset contains 1,910 automatically generated questions about three rooms and 90 objects that have been annotated for their class, location, colour, and spatial relationships to other objects. 

# Install dependencies

`pip install -r requirements.txt`

# Data

The raw image data can be found in `\images`. The annotations of these images and of the objects that occur on those images, are stored in `\annotations\annotations.csv`. The questions can be found in `\questions\questions_downsampled.csv`. 

# 
