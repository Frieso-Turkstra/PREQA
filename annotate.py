"""
Author: Frieso Turkstra
Date: 2024-06-14

This program ...
"""

from PIL import ImageTk, ImageDraw, Image
from tkinter import messagebox, ttk
from pathlib import Path
import tkinter as tk
import pandas as pd
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file",
                        help="The preprocessed and resolved annotations file.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_file",
                        help="Path to the output file.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args


class App(tk.Tk):
    def __init__(self, input_file):
        super().__init__()
        self.objects = pd.read_csv(input_file)
        self.num_objects = len(self.objects["object_id"].unique())
        self.object_id = -1

        self.init_ui()

    def init_ui(self):
        self.title("Object Attribute Annotation")
        self.state("zoomed")

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
        self.location_combobox["state"] = "readonly"  # Make the combobox read-only
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
        self.on_label = tk.Label(self.prepositions_frame, text="on")
        self.on_label.pack(side="left")
        self.on_entry = tk.Entry(self.prepositions_frame)
        self.on_entry.pack(side="left")

        self.above_label = tk.Label(self.prepositions_frame, text="above")
        self.above_label.pack(side="left")
        self.above_entry = tk.Entry(self.prepositions_frame)
        self.above_entry.pack(side="left")

        self.below_label = tk.Label(self.prepositions_frame, text="below")
        self.below_label.pack(side="left")
        self.below_entry = tk.Entry(self.prepositions_frame)
        self.below_entry.pack(side="left")

        self.next_to_label = tk.Label(self.prepositions_frame, text="next_to")
        self.next_to_label.pack(side="left")
        self.next_to_entry = tk.Entry(self.prepositions_frame)
        self.next_to_entry.pack(side="left")

        # Create a submit button.
        self.save_button = tk.Button(self, text="Next", command=self.save_annotations)
        self.save_button.pack()
        self.bind("<Return>", lambda _: self.save_annotations())

        # Keep track of the number of annotated objects.
        self.count_label = tk.Label(self)
        self.count_label.pack()

        self.next_image()

    def get_image(self, obj):
        # Open image and draw the bounding box
        image_directory = Path("images")
        image_file_path = image_directory / obj.filename
        image = Image.open(image_file_path)
        draw = ImageDraw.Draw(image)
        draw.rectangle([obj.x1, obj.y1, obj.x2, obj.y2], outline="red", width=3)

        return ImageTk.PhotoImage(image)
    
    def save_annotations(self):
        if self.save_location() and self.save_colours() and self.save_prepositions():
            self.colour_entry.delete(0, tk.END)
            self.on_entry.delete(0, tk.END)
            self.above_entry.delete(0, tk.END)
            self.below_entry.delete(0, tk.END)
            self.next_to_entry.delete(0, tk.END)
            self.next_image()

    def save_location(self):
        location = self.location_combobox.get()
        if not location:
            messagebox.showinfo("Error", "Please choose a location.")
            return False
        objects = self.objects.loc[self.objects["object_id"] == self.object_id, "location"] = location
        return True

    def save_colours(self):
        colours = self.colour_entry.get()
        if not colours:
            messagebox.showinfo("Error", "Please enter one or more colours.")
            return False
        colours = [colour.lower().strip() for colour in colours.split(",")]

        colours_df = pd.read_csv("resources/colours.csv")
        possible_colours = colours_df["colour_name"].unique().tolist()
        for colour in colours:
            if colour not in possible_colours:
                messagebox.showinfo("Error", f"Invalid colour: {colour}")
                return False

        self.objects.loc[self.objects["object_id"] == self.object_id, "colours"] = str(colours)
        return True
    
    def save_prepositions(self):
        on = [prep.lower().strip() for prep in self.on_entry.get().split(",")]
        above = [prep.lower().strip() for prep in self.above_entry.get().split(",")]
        below = [prep.lower().strip() for prep in self.below_entry.get().split(",")]
        next_to = [prep.lower().strip() for prep in self.next_to_entry.get().split(",")]

        possible_objects = [str(i) for i in range(self.num_objects) if i != self.object_id]
        for obj in on:
            if obj and obj not in possible_objects:
                messagebox.showinfo("Error", f"Non-existing object: {obj}")
                return False
            
        for obj in above:
            if obj and obj not in possible_objects:
                messagebox.showinfo("Error", f"Non-existing object: {obj}")
                return False
            
        for obj in below:
            if obj and obj not in possible_objects:
                messagebox.showinfo("Error", f"Non-existing object: {obj}")
                return False
            
        for obj in next_to:
            if obj and obj not in possible_objects:
                messagebox.showinfo("Error", f"Non-existing object: {obj}")
                return False
            
        on = list(map(int, on))
        above = list(map(int, above))
        below = list(map(int, below))
        next_to = list(map(int, next_to))

        self.objects.loc[self.objects["object_id"] == self.object_id, "on"] = str(on) if on[0] else "[]"
        self.objects.loc[self.objects["object_id"] == self.object_id, "above"] = str(above) if above[0] else "[]"
        self.objects.loc[self.objects["object_id"] == self.object_id, "below"] = str(below) if below[0] else "[]"
        self.objects.loc[self.objects["object_id"] == self.object_id, "next_to"] = str(next_to) if next_to[0] else "[]"
        return True
  
    def next_image(self):
        self.object_id += 1
        self.count_label.config(text=f"{self.object_id}/{self.num_objects}")
    
        if self.object_id >= self.num_objects:
            messagebox.showinfo("Info", "No more images to annotate.")
            print("You have annotated this many objects:", self.object_id)
            self.quit()
            return
        
        obj = None
        possible_obj = self.objects.loc[self.objects["object_id"] == self.object_id]
        for occlusion in ("none", "partial", "severe"):
            objects = possible_obj.loc[possible_obj["occlusion"] == occlusion]
            if not objects.empty:
                obj = objects.iloc[0]
                break
        
        image = self.get_image(obj)

        self.image_label.config(image=image)
        self.image_label.image = image

        self.file_label.config(text=obj.filename)


    def save_file(self, output_file_path):
        self.objects.to_csv(output_file_path, index=False)


def main():
    # Read in the command-line arguments.
    args = create_arg_parser()

    # Ensure the input file is a csv file that exists.
    input_file = Path(args.input_file)
    if not (input_file.exists() and input_file.is_file() and input_file.suffix.lower() == ".csv"):
        raise FileNotFoundError(
            f"The file '{input_file}' does not exist or is not a CSV file."
            )
    
    # Start the graphical user interface, save the results when it is closed.
    app = App(input_file)
    app.mainloop()
    app.save_file(args.output_file)


if __name__ == "__main__":
    main()
