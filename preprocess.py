"""
Author: Frieso Turkstra
Date: 2024-06-14

This program takes as input a csv file from the VGG Image Annotator (VIA).
Ensure each region has two attributes: 'label' and 'occlusion'.
Furthermore, all bounding boxes are assumed to be rectangular.

The following preprocessing steps are taken:

- Discard images that contain no objects with bounding boxes.
- Assign a unique identifier to each region.
- Flatten columns that contain dictionaries into separate columns.
- Change the bounding box notation so it works with the Pillow library.
- Add WordNet senses based on the labels.
- Remove unnecessary columns.
"""

from pathlib import Path
import pandas as pd
import argparse
import uuid
import json


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file",
                        help="File with the VIA annotations.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_file",
                        help="Path to the output file.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args


def main():
    # Read in the command-line arguments.
    args = create_arg_parser()

    # Ensure the input file is a csv file that exists.
    input_file = Path(args.input_file)
    if not (input_file.exists() and input_file.is_file() and input_file.suffix.lower() == ".csv"):
        raise FileNotFoundError(
            f"The file '{input_file}' does not exist or is not a CSV file."
            )

    # Load the annotations into a dataframe.
    df = pd.read_csv(args.input_file)

    # Filter images that contain no objects with bounding boxes.
    df = df[df["region_count"] > 0]

    # Assign a unique identifier to each region.
    df["uid"] = [uuid.uuid4() for _ in range(len(df))]

    # Flatten the region_shape_attributes column.
    # Assumes all objects are annotated with rectangular bounding boxes.
    # The bounding box notation is changed to the one used by the Pillow libary.
    bounding_boxes = df["region_shape_attributes"].apply(json.loads)
    df["x1"] = bounding_boxes.apply(lambda bbox: bbox["x"])
    df["y1"] = bounding_boxes.apply(lambda bbox: bbox["y"])
    df["x2"] = bounding_boxes.apply(lambda bbox: bbox["x"] + bbox["width"])
    df["y2"] = bounding_boxes.apply(lambda bbox: bbox["y"] + bbox["height"])

    # Flatten the region_attributes column.
    # Assumes each region has been assigned a label and a level of occlusion.
    region_attributes = df["region_attributes"].apply(json.loads)
    df["label"] = region_attributes.apply(lambda attrs: attrs["label"])
    df["occlusion"] = region_attributes.apply(lambda attrs: attrs["occlusion"])

    # Add WordNet senses.
    classes = pd.read_csv("resources/classes.csv")
    mappings = dict(zip(classes["class"], classes["wordnet_sense"]))
    df["wordnet_sense"] = df["label"].apply(lambda x: mappings[x])

    # Remove unnecessary columns.
    unnecessary_columns = [
        "file_size",
        "file_attributes",
        "region_shape_attributes",
        "region_attributes",
    ]
    df.drop(unnecessary_columns, axis=1, inplace=True)

    # Save the new preprocessed csv file to the specified output file.
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
