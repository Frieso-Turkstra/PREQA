"""
Author: Frieso Turkstra
Date: 2024-06-15

This program implements a graphical user interface to help with the annotation
for object attributes. You can annotate the following attributes:

- Location
- Colour
- Spatial relationships (on, above, below, lower)

Location and colour are necessary fields. Spatial relationships are optional.
Annotate spatial relationships by specifying the object ids.
Multiple colours and objects are allowed if separated by commas.
"""

from PIL import ImageTk, ImageDraw, Image
from tkinter import messagebox, ttk
from pathlib import Path
import tkinter as tk
import pandas as pd
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations_file",
                        help="The preprocessed and resolved annotations file.",
                        required=True,
                        type=str)
    parser.add_argument("-i", "--image_directory",
                        help="Directory where the images are stored.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_file",
                        help="Path to the output file.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args


class App(tk.Tk):
    def __init__(self, annotations_file, image_directory):
        super().__init__()
        self.objects = pd.read_csv(annotations_file)
        self.image_directory = image_directory
        self.num_objects = len(self.objects["object_id"].unique())
        self.object_id = -1

        # Initialise the user interface and show the first image for annotation.
        self.init_ui()
        self.next_image()

    def init_ui(self):
        self.title("Object Attribute Annotation")
        self.state("zoomed")

        # Create two labels, one for the image and one for the image's filename.
        self.image_label = tk.Label(self)
        self.image_label.pack()

        self.file_label = tk.Label(self)
        self.file_label.pack()

        # Create a drop-down menu for the locations.
        rooms_df = pd.read_csv("resources/rooms.csv")
        locations = rooms_df["room_name"].unique().tolist()
        selected_location = tk.StringVar()

        self.location_combobox = ttk.Combobox(self, textvariable=selected_location)
        self.location_combobox["values"] = locations
        self.location_combobox["state"] = "readonly"
        self.location_combobox.pack(pady=10)

        # Create an entry for the colours.
        self.colour_frame = tk.Frame(self, pady=10)
        self.colour_frame.pack()

        self.colour_label = tk.Label(self.colour_frame, text="Colour")
        self.colour_label.pack(side="left")
        self.colour_entry = tk.Entry(self.colour_frame)
        self.colour_entry.pack(side="left")

        # Create four entries for the prepositions.
        self.prepositions_frame = tk.Frame(self, pady=10)
        self.prepositions_frame.pack()

        self.prepositions = ["on", "above", "below", "next_to"]

        self.preposition_entries = {}
        for preposition in self.prepositions:
            label = tk.Label(self.prepositions_frame, text=preposition)
            label.pack(side="left")
            entry = tk.Entry(self.prepositions_frame)
            entry.pack(side="left")
            self.preposition_entries[preposition] = entry

        # Create a button to save the annotations.
        self.save_button = tk.Button(self, text="Next", command=self.save_annotations)
        self.save_button.pack()
        self.bind("<Return>", lambda _: self.save_annotations())

        # Keep track of the number of annotated objects.
        self.count_label = tk.Label(self)
        self.count_label.pack()

    def save_annotations(self):
        # If all the necessary inputs are provided and valid,
        if self.save_location() and self.save_colours() and self.save_prepositions():

            # reset the entries,
            self.colour_entry.delete(0, tk.END)
            for entry in self.preposition_entries.values():
                entry.delete(0, tk.END)

            # and show the next image to be annotated.
            self.next_image()
    
    def save_location(self):
        # Check if a location has been chosen.
        location = self.location_combobox.get()
        if not location:
            messagebox.showinfo("Error", "Please choose a location.")
            return False
        
        # Otherwise, save the location in the annotation file.
        self.objects.loc[self.objects["object_id"] == self.object_id, "location"] = location
        return True

    def save_colours(self):
        # Check if at least one colour has been entered.
        colours = self.colour_entry.get()
        if not colours:
            messagebox.showinfo("Error", "Please enter one or more colours.")
            return False
        
        # Clean and format the user input.
        colours = [colour.lower().strip() for colour in colours.split(",")]

        # Get a list of all valid colour options.
        colours_df = pd.read_csv("resources/colours.csv")
        possible_colours = colours_df["colour_name"].unique().tolist()

        # Check if the colours provided by the user are valid colours.
        for colour in colours:
            if colour not in possible_colours:
                messagebox.showinfo("Error", f"Invalid colour: {colour}")
                return False

        # Otherwise, save the colours in the annotation file.
        self.objects.loc[self.objects["object_id"] == self.object_id, "colour"] = str(colours)
        return True
    
    def save_prepositions(self):
        # Clean and format the user input.
        objects = {}
        for preposition, entry in self.preposition_entries.items():
            objects[preposition] = [obj.lower().strip() for obj in entry.get().split(",") if obj]

        # Get a set with all valid object ids.
        possible_objects = {str(i) for i in range(self.num_objects) if i != self.object_id}

        # Check if the object ids provided by the user are valid object ids.
        for objs in objects.values():
            if not set(objs) <= (possible_objects):
                messagebox.showinfo("Error", f"Found the object itself or a non-existing object in {objs}.")
                return False

        # All object ids are now valid so the conversion to integers is safe.
        objects = {preposition: list(map(int, objs)) for preposition, objs in objects.items()}

        # Save the spatial relationships in the annotation file.
        for preposition, objs in objects.items():
            self.objects.loc[self.objects["object_id"] == self.object_id, preposition] = str(objs)
        return True
    
    def get_image(self, obj):
        # Open the image and draw the bounding box.
        image_file_path = Path(self.image_directory / obj.filename)
        image = Image.open(image_file_path)
        draw = ImageDraw.Draw(image)
        draw.rectangle([obj.x1, obj.y1, obj.x2, obj.y2], outline="red", width=3)

        # Convert to a tkinter image.
        return ImageTk.PhotoImage(image)
  
    def next_image(self):
        # Move to the next object id and update the count.
        self.object_id += 1
        self.count_label.config(text=f"{self.object_id}/{self.num_objects}")
        
        # Check if there are any annotations left to do.
        if self.object_id >= self.num_objects:
            messagebox.showinfo("Info", "No more images to annotate.")
            print("You have annotated this many objects:", self.object_id)
            self.quit()
            return
        
        # Find the image with the best possible view of the object.
        views = self.objects.loc[self.objects["object_id"] == self.object_id]

        for occlusion in ("none", "partial", "severe"):
            if not (objects := views.loc[views["occlusion"] == occlusion]).empty:
                obj = objects.iloc[0]
                break
        
        # Convert to a tkinter image and draw the bounding box on the image.
        image = self.get_image(obj)

        # Update the image and filename labels.
        self.image_label.config(image=image)
        self.image_label.image = image
        self.file_label.config(text=obj.filename)


    def save_file(self, output_file_path):
        self.objects.to_csv(output_file_path, index=False)


def main():
    # Read in the command-line arguments.
    args = create_arg_parser()

    # Ensure the input file is a csv file that exists.
    annotations_file = Path(args.annotations_file)
    if not (
        annotations_file.exists() and
        annotations_file.is_file() and
        annotations_file.suffix.lower() == ".csv"
    ):
        raise FileNotFoundError(
            f"The file '{annotations_file}' does not exist or is not a CSV file."
            )
    
    # Ensure the image directory exists.
    image_directory = Path(args.image_directory)
    if not (image_directory.exists() and image_directory.is_dir()):
        raise FileNotFoundError(
            f"The directory '{image_directory}' does not exist."
            )
    
    # Start the graphical user interface, save the results when it is closed.
    app = App(annotations_file, image_directory)
    app.mainloop()
    app.save_file(args.output_file)


if __name__ == "__main__":
    main()
