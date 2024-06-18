"""
Author: Frieso Turkstra
Date: 2024-06-18

This program takes as input a directory of images.
The image filenames are in the following format: viewpoint-direction-tilt.jpg
viewpoint is a number between 0 and n where n is the total number of viewpoints.
direction is a number where 0 = North, 1 = East, 2 = South, 3 = West.
tilt is a number where 0 = -25 degrees, 1 = 0 degrees, and 2 = 25 degrees tilt.

Additionally, a csv file must be provided that specifies which viewpoints are 
connected to which viewpoints in which directions. The program can then
automatically load the images into a navigable graph notation.
"""

import matplotlib.pyplot as plt 
from pathlib import Path
from PIL import Image
import networkx as nx
import pandas as pd
import itertools
import argparse

MAX_NUM_ACTIONS = 100

# TODO
# Skip for now, first focus on if it is even necessary.
# Translate images to actions (0-1-0.jpg --> 0-2-0.jpg = rotate_right)
# Translate actions to images (0-1-0.jpg + rotate-right = 0-2-0.jpg)

class Node:
    def __init__(self, image_file_path):
        self.view = Image.open(image_file_path)
        self.image_file_path = Path(image_file_path)
        self.viewpoint, self.direction, self.tilt = map(
            int, self.image_file_path.stem.split("-")
            )


class Environment:
    def __init__(self, image_files: [str]):
        # Read in the data from the image files.
        self.nodes = [Node(file) for file in image_files]
        self.num_viewpoints = len(set([node.viewpoint for node in self.nodes]))
        self.num_directions = len(set([node.direction for node in self.nodes]))
        self.num_tilts = len(set([node.tilt for node in self.nodes]))

        # Construct the graph.
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(self.get_edges()) # automatically adds nodes
        print(self.graph.number_of_nodes())
        print(self.graph.number_of_edges())

    def get_edges(self) -> [(Node,Node)]:
        all_possible_edges = itertools.permutations(self.nodes, 2)
        df = pd.read_csv("resources/edges.csv")

        edges = []
        for node1, node2 in all_possible_edges:
            # Tilt neighbours.
            if node1.viewpoint == node2.viewpoint and node1.direction == node2.direction and abs(node1.tilt - node2.tilt) == 1:
                edges.append((node1, node2))

            # Direction neighbours.
            if node1.viewpoint == node2.viewpoint and abs(node1.direction - node2.direction) != 2 and node1.tilt == node2.tilt:
                edges.append((node1, node2))

            # Viewpoint neighbours.
            if node1.direction == node2.direction and node1.tilt == node2.tilt:
                if not df[
                    (df["source"] == node1.viewpoint) &
                    (df["target"] == node2.viewpoint) &
                    (df["direction"] == node1.direction)].empty:
                    edges.append((node1, node2))

        return edges

    def show_graph(self) -> None:
        pos = nx.spring_layout(self.graph)  # positions for all nodes
        nx.draw(self.graph, pos, with_labels=False, node_color='skyblue', node_size=100, font_size=15, font_color='black')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels={(u, v): '' for u, v in self.graph.edges()})
        plt.show()

    def get_shortest_path(self, source, target) -> [Node]:
        return nx.shortest_path(self.graph, source=source, target=target)

    # def nodes2action(self, source, target):
    #     if not self.graph.has_edge(source, target):
    #         print("Could not translate the nodes two an action: no common edge found.")
    #         return

    #     if source.viewpoint == target.viewpoint and source.direction == target.direction and abs(source.tilt - target.tilt) == 1:
    #         return "tilt_down" if source.tilt > target.tilt else "tilt_up" 

    #     if source.viewpoint == target.viewpoint and abs(source.direction - target.direction) != 2 and source.tilt == target.tilt:
    #         delta = target.direction - source.direction
    #         if difference == 1 or difference == -3:
    #             return "right"
    #         elif difference == -1 or difference == 3:
    #             return "left"
    #         else:
    #             # This covers cases where the difference is 0 (same direction) or 2 (U-turn)
    #             # Should not be possible due to the initial has_neighbour check.
    #             return "no turn" 

        # Also check for forward!!

        




def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_directory",
                        help="Directory with the images of the environment.",
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
    
    # Generate graph, navigate environment, save stuff.
    env = Environment(image_files)
    node1 = env.nodes[0]
    node2 = env.nodes[3]
    print("Node 1: ", vars(node1))
    print("Node 2: ", vars(node2))
    shortest_path = env.get_shortest_path(node1, node2)
    print(len(shortest_path))

    for node in shortest_path:
        print(node.image_file_path)

if __name__ == "__main__":
    main()
