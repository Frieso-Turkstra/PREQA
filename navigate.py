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

1) An environment is constructed as a graph of images.
2) A robot is initialized with a navigator (random, shortest_path, optimal_view_path)
3) The robot explores the environment until it stops or max_num_actions.
4) The path is saved and evaluated.
"""

import matplotlib.pyplot as plt 
from pathlib import Path
from PIL import Image
import networkx as nx
import pandas as pd
import itertools
import argparse
import random


class Node:
    def __init__(self, image_file: Path):
        self.image_file = image_file
        self.viewpoint, self.direction, self.tilt = map(int, self.image_file.stem.split("-"))

    def __repr__(self):
        return f"Node<{self.viewpoint}, {self.direction}, {self.tilt}>"


class Environment:
    def __init__(self, image_files: [str], annotations_file):
        # Read the annotation file.
        self.annotations = pd.read_csv(annotations_file)

        # Load the image files into nodes.
        nodes = [Node(file) for file in image_files]

        # Infer the number of viewpoints, directions and tilts.
        self.num_viewpoints = max(node.viewpoint for node in nodes) + 1
        self.num_directions = max(node.direction for node in nodes) + 1
        self.num_tilts = max(node.tilt for node in nodes) + 1

        # Create the edges.
        edges = self.get_edges(nodes)

        # Construct an unweighted, directed graph from the nodes and edges.
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges) # automatically adds the nodes
    
    def get_edges(self, nodes) -> [(Node,Node)]:
        # Note: this function is specifically designed to work for num_tilt = 3, num_directions = 4.
        all_possible_edges = itertools.permutations(nodes, 2)
        df = pd.read_csv("resources/edges.csv")

        edges = []
        for node1, node2 in all_possible_edges:
            # Tilt neighbours.
            if (
                node1.viewpoint == node2.viewpoint and
                node1.direction == node2.direction and
                abs(node1.tilt - node2.tilt) == 1
                ):
                edges.append((node1, node2))

            # Direction neighbours.
            if (
                node1.viewpoint == node2.viewpoint and
                abs(node1.direction - node2.direction) != 2 and
                node1.tilt == node2.tilt
                ):
                edges.append((node1, node2))

            # Viewpoint neighbours.
            if node1.direction == node2.direction and node1.tilt == node2.tilt:
                if not df[
                    (df.source == node1.viewpoint) &
                    (df.target == node2.viewpoint) &
                    (df.direction == node1.direction)].empty:
                    edges.append((node1, node2))

        return edges

    def show_graph(self) -> None:
        pos = nx.spring_layout(self.graph)
        nx.draw(
            self.graph, pos, with_labels=False, node_color="skyblue",
            node_size=100, font_size=15, font_color="black"
            )
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels={(u, v): "" for u, v in self.graph.edges()}
            )
        plt.show()

    def shortest_path(self, source: Node, target:Node) -> [Node]:
        return nx.shortest_path(self.graph, source=source, target=target)

    def shortest_path_to_object(self, source: Node, object_id: int) -> [Node]:
        # Find all images that display the object.
        filenames = self.annotations[self.annotations.object_id == object_id].filename
        
        # Find the nodes corresponding to the images.
        nodes = []
        for filename in filenames:
            for node in self.graph.nodes():
                if node.image_file.name == filename:
                    nodes.append(node)
                    break

        shortest_paths = [self.shortest_path(source, node) for node in nodes]
        return min(shortest_paths, key=len)

    def occlusion_sort(self, obj) -> tuple:
        occlusion_levels = {'none': 0, 'partial': 1, 'severe': 2}
        object_area = (obj.x2 - obj.x1) * (obj.y2 - obj.y1)
        return (occlusion_levels[obj.occlusion], -object_area)

    def shortest_path_to_optimal_view(self, source: Node, object_id: int) -> [Node]:
        # Find all images that display the object.
        images = self.annotations[self.annotations.object_id == object_id]

        # Sort them by level of occlusion: none < partial < severe.
        # If same level of occlusion, choose the largest bounding box.
        sorted_images = sorted(images.iterrows(), key=lambda x: self.occlusion_sort(x[1]))
        optimal_view = sorted_images[0][1].filename

        # Find the node corresponding to the optimal view.
        the_node = None
        for node in self.graph.nodes():
            if node.image_file.name == optimal_view:
                the_node = node
                break

        shortest_path = self.shortest_path(source, the_node)
        return shortest_path
 
    def are_connected(self, source: Node, target: Node) -> bool:
        return target in self.graph.successors(source)

    def evaluate(self, target, path: [Node]) -> dict[str, int]:
        source = path[0]
        # Distance to the target object at termination (DT).
        final_distance_to_target = len(self.shortest_path(path[-1], target)) - 1

        # Change in distance to the target object from the initial and final position (D∆).
        initial_distance_to_target = len(self.shortest_path(source, target)) - 1
        change_in_distance_to_target = initial_distance_to_target - final_distance_to_target

        # Smallest distance to the target object at any point in the navigation (Dmin)
        minimal_distance_to_target = min(
            len(self.shortest_path(node, target)) - 1 for node in path
        )

        return {
            "final_distance_to_target": final_distance_to_target,
            "change_in_distance_to_target": change_in_distance_to_target,
            "minimal_distance_to_target": minimal_distance_to_target,
        }

    def get_starting_position(self, num_actions, target):
        all_shortest_paths = []
        for node in self.graph.nodes():
            shortest_path = self.shortest_path(node, target)
            if len(shortest_path) - 1 == num_actions:
                return node
        print(f"No node is {num_actions} actions removed from {target}.")

    def random_path(self, source):
        num_actions = 10
        path = [source]
        for i in range(num_actions):
            neighbours = list(self.graph.successors(path[-1]))
            path.append(random.choice(neighbours))
        return path

    def forward_only_path(self, source):
        num_actions = 10
        path = [source]
        for i in range(num_actions):
            neighbours = list(self.graph.successors(path[-1]))
            for neighbour in neighbours:
                if neighbour.direction == path[-1].direction and neighbour.tilt == path[-1].tilt:
                    # Go forward
                    path.append(neighbour)
                    break
        return path

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations_file",
                        help="The finished annotations file.",
                        required=True,
                        type=str)
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

    # Ensure the input directory already exists.
    input_directory = Path(args.input_directory)
    if not (input_directory.exists() and input_directory.is_dir()):
        raise FileNotFoundError(
            f"The directory '{input_directory}' does not exist."
            )

    # Select only image files from the input directory.
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [
        Path(file) for file in input_directory.iterdir()
        if file.suffix.lower() in image_extensions
    ]
    
    # Create environment.
    env = Environment(image_files, annotations_file)

    # Pick a target.
    target = random.choice(list(env.graph.nodes()))

    # Start the robot N actions away from target.
    nodes = list(env.graph.nodes())
    node1 = nodes[12]
    path = env.forward_only_path(node1)

    print(path)
    print(env.evaluate(target, path))

    # Save path to output file and the type of navigation and the question?


if __name__ == "__main__":
    main()
