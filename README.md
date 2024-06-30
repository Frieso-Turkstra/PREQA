# PREQA: A Photorealistic Dataset for Embodied Question Answering

PREQA is a high-quality dataset that is designed to bring the task of Embodied Question Answering (EQA) as close as possible to the real-world. As such, its
primary focus is to be used as a test set for pre-trained EQA systems. The dataset consists of images of a real-world robot lab and introduces new features such as camera tilt, viewpoint-based navigation and novel annotations. The dataset contains 1,910 automatically generated questions about three rooms and 90 objects that have been annotated for their class, location, colour, and spatial relationships to other objects. 

The raw image data, annotations and questions can be found in `\PREQA`. In the `\resources` directory, you can find the possible object classes, room types and colour values as well as the viewpoints data and the question templates. 

If you want to create your own EQA dataset using the PREQA-method, follow the instructions down below.

# How to create your own PREQA dataset

Tip: Use `name_of_python_script.py --help` to see how to run a specific Python script.

 1. **Install dependencies**: Run `pip install -r requirements.txt`.
 2. **Data collection**: Decide on the number of viewpoints and their location. From each viewpoint, take pictures in all directions - depending on the desired rotational degrees of freedom and the number of camera tilts. 
3. **Annotation**: Identify all the bounding boxes using image annotation software such as [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/via.html). Annotate each bounding box for its level of occlusion and the class label of the depicted object. Use `annotations/preprocess.py` to preprocess the VIA annotations. Then, use `annotations/resolve.py` to identify all unique object instances and `annotations/annotate.py` to annotate each object instance for its colour, location and spatial relationships to other objects.
4. **Question generation**: Run `questions/questions.py` to automatically generate the questions from the annotations file. Use `questions/downsample.py` to balance the dataset. 
5. **Navigation**: Create a csv file which outlines how the viewpoints are connected to each other. See `resources/edges.csv` for reference. Input this file with the edges into `navigation/navigate.py` to navigate the environment. Either use one of the existing navigation modules (random, forward only, shortest path to object, shortest path to optimal view) or modify the code to implement your own.
6. You have now created a new EQA dataset based on a real-world environment!
