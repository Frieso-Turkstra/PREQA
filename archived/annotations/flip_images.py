"""
Author: Frieso Turkstra
Date: 2024-06-14

This program takes as input a directory with image files,
flips them horizontally, and saves them to an output directory.
"""

from pathlib import Path
from PIL import Image
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory",
                        help="Directory with the images to be flipped.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_directory",
                        help="Directory to which the flipped images are saved.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args


def main():
    # Read in the command-line arguments.
    args = create_arg_parser()

    # Create the output directory if it does not exist yet.
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Ensure the input directory already exists.
    input_directory = Path(args.input_directory)
    if not (input_directory.exists() and input_directory.is_dir()):
        raise FileNotFoundError(
            f"The directory '{input_directory}' does not exist."
            )
    
    # Select only image files from the input directory.
    image_extensions = {".jpg", ".jpeg", ".png"}   
    image_files = [
        file for file in input_directory.iterdir()
        if file.suffix.lower() in image_extensions
        ]
        
    # Flip all the images and save them to the output directory.
    for image_file in image_files:
        with Image.open(image_file) as image:
            flipped_image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
            flipped_image.save(output_directory / image_file.name)


if __name__ == "__main__":
    main()
