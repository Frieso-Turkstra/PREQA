"""
Author: Frieso Turkstra
Date: 2024-06-14

This program implements a graphical user interface to help with the annotation
for instance-level recognition. It generates all pairs of objects with the same
label and presents them to the user. The user then marks the objects as the same
or different using either the buttons or the hotkeys (s, d). A Union-Find
algorithm is used to reduce the number of necessary comparisons by inferring
which comparisons are redundant based on the already made comparisons.
"""

from PIL import Image, ImageTk, ImageDraw
from tkinter import messagebox
from pathlib import Path
import tkinter as tk
import pandas as pd
import itertools
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations_file",
                        help="File with the preprocessed VIA annotations.",
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


class UnionFind:
    def __init__(self, objects):
        # Each object starts as a disjoint set, meaning the object is
        # its own parent and has rank 0.
        self.parent = {obj: obj for obj in objects}
        self.rank = {obj: 0 for obj in objects}

    def find(self, obj):
        # Find the root parent of obj. Use path compression (updating the parent
        # of obj as we move up the tree) to ensure fast look-up times.
        if self.parent[obj] != obj:
            self.parent[obj] = self.find(self.parent[obj])
        return self.parent[obj]

    def union(self, pair):
        # Find the root members of the sets to which obj1 and obj2 belong.
        obj1, obj2 = pair
        root1 = self.find(obj1)
        root2 = self.find(obj2)

        # Try to union the smaller tree (i.e. the one with the lower rank)
        # into the larger tree. If they are of equal size, it does not matter.
        # This ensures the tree structure stays flat and therefore, efficient.
        if root1 != root2:
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


class App(tk.Tk):
    def __init__(self, annotations_file, image_directory):
        super().__init__()
        self.regions = pd.read_csv(annotations_file)
        self.image_directory = image_directory
        self.num_comparisons = 0
        self.index = -1

        # Generate all the pairs that need comparing.
        self.region_pairs = self.get_region_pairs()

        # Create a Union-Find data structure with the regions.
        self.union_find = UnionFind(self.regions.uid)

        # Initialize the user interface.
        self.init_ui()

    def get_region_pairs(self):
        # Create all possible combinations of objects that have the same label.
        # A comparison between 2 objects is represented as a pair of their uids.
        # E.g. {A, B, C} --> (A, B) & (A, C) & (B, C)
        pairs = []
        for label in self.regions["label"].unique():
            uids = self.regions.loc[self.regions["label"] == label, "uid"]
            pairs += list(itertools.combinations(uids, 2))
        return pairs

    def init_ui(self):
        self.title("Annotation for Instance-Level Recognition")
        self.state("zoomed")

        # Create labels to hold the images.
        self.image_label_1 = tk.Label(self)
        self.image_label_1.grid(row=0, column=0)

        self.image_label_2 = tk.Label(self)
        self.image_label_2.grid(row=0, column=1)
        
        # Create labels to hold the filenames of the images.
        self.file_label_1 = tk.Label(self)
        self.file_label_1.grid(row=1, column=0)

        self.file_label_2 = tk.Label(self)
        self.file_label_2.grid(row=1, column=1)

        # Create two buttons to mark the regions as different or identical.
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=2, columnspan=2)
        
        self.same_button = tk.Button(self.button_frame, text="Same", command=self.mark_same)
        self.same_button.grid(row=0, column=0, padx=10, pady=10)
        self.bind("<s>", lambda _: self.mark_same())

        self.different_button = tk.Button(self.button_frame, text="Different", command=self.mark_different)
        self.different_button.grid(row=0, column=1, padx=10, pady=10)
        self.bind("<d>", lambda _: self.mark_different())

        # Create a label to keep track of how many comparisons are left.
        self.count_label = tk.Label(self, text=f"{self.index}/{len(self.region_pairs)}")
        self.count_label.grid(row=3, columnspan=2)

        self.next_pair()

    def mark_same(self):
        # Union the sets of the current objects and continue to the next pair.
        current_pair = self.region_pairs[self.index]
        self.union_find.union(current_pair)
        self.num_comparisons += 1
        self.next_pair()

    def mark_different(self):
        # Continue to the next pair.
        self.num_comparisons += 1
        self.next_pair()

    def get_image(self, region):
        # Open the region's image and draw its bounding box.
        image_file_path = Path(self.image_directory / region.filename)
        image = Image.open(image_file_path)
        draw = ImageDraw.Draw(image)
        draw.rectangle([region.x1, region.y1, region.x2, region.y2], outline="red", width=3)

        # Resize the image to fit half the screen.
        max_width = self.winfo_screenwidth() // 2
        width_percent = (max_width / float(image.size[0]))
        new_height = int((float(image.size[1]) * float(width_percent)))
        image = image.resize((max_width, new_height), Image.LANCZOS)

        # Convert to a tkinter image.
        return ImageTk.PhotoImage(image)

    def next_pair(self):
        # Update the count label.
        self.index += 1
        self.count_label.config(text=f"{self.index}/{len(self.region_pairs)}")

        # Check if we have any comparisons left to do.
        if self.index >= len(self.region_pairs):
            messagebox.showinfo("Info", "No more images to compare.")
            self.quit()
            return
        
        # Get the regions of the current comparison.
        uid1, uid2 = self.region_pairs[self.index]
        region1 = self.regions.loc[self.regions["uid"] == uid1].squeeze()
        region2 = self.regions.loc[self.regions["uid"] == uid2].squeeze()

        # We can skip all comparisons with non-root regions.
        if self.union_find.parent[uid1] != uid1 or self.union_find.parent[uid2] != uid2:
            self.next_pair()
            return
        
        # Comparison is necessary, show the two regions on the images.    
        img1_tk = self.get_image(region1)
        img2_tk = self.get_image(region2)

        # Update the image and file labels.
        self.image_label_1.config(image=img1_tk)
        self.image_label_2.config(image=img2_tk)
        
        self.image_label_1.image = img1_tk
        self.image_label_2.image = img2_tk

        self.file_label_1.config(text=region1.filename)
        self.file_label_2.config(text=region2.filename)

    def save_file(self, output_file_path):
        # Map each region to its root.
        region_to_root = {uid: self.union_find.find(uid) for uid in self.regions.uid}

        # Assign an id to each group.
        unique_roots = set(region_to_root.values())
        root_to_id = {root: i for i, root in enumerate(unique_roots)}

        # Map each region to its group id.
        region_to_group_id = {uid: root_to_id[region_to_root[uid]] for uid in self.regions.uid}
        
        # Create a new column to store each region's group id.
        self.regions["object_id"] = None
        for uid, object_id in region_to_group_id.items():
            self.regions.loc[self.regions["uid"] == uid, "object_id"] = object_id
        
        # Save the results.
        self.regions.to_csv(output_file_path, index=False)
        print(f"You performed {self.num_comparisons} comparisons!")


def main():
    # Read in the command-line arguments.
    args = create_arg_parser()

    # Ensure the annotations file is a csv file that exists.
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
