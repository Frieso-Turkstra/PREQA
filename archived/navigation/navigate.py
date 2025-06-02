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

The program generates paths for each of the following navigation strategies:
- random
- forward only
- shortest path to object
- shortest path to optimal view
The paths are sets of images. Each path is turned into a single image by collecting
each image into a grid. These stitched images can be used to prompt a single-image
VLM (vision language model).
"""

from networkx.algorithms.approximation import traveling_salesman_problem, greedy_tsp
import matplotlib.pyplot as plt 
from functools import lru_cache
from pathlib import Path
from PIL import Image
import networkx as nx
import pandas as pd
import itertools
import argparse
import random
import math
import os


seed_value = 42
random.seed(seed_value)


class Node:
    def __init__(self, image_file: Path):
        # Each node is an image that has a viewpoint, direction and tilt.
        self.image_file = image_file
        self.viewpoint, self.direction, self.tilt = map(int, self.image_file.stem.split("-"))

    def __repr__(self):
        return f"Node<{self.viewpoint}, {self.direction}, {self.tilt}>"


class Environment:
    def __init__(self, image_files, annotations_file, edges_file):
        # Read the annotation file.
        self.annotations = pd.read_csv(annotations_file)

        # Load the image files into nodes.
        nodes = [Node(file) for file in image_files]

        # Infer the number of viewpoints, directions and tilts.
        self.num_viewpoints = max(node.viewpoint for node in nodes) + 1
        self.num_directions = max(node.direction for node in nodes) + 1
        self.num_tilts = max(node.tilt for node in nodes) + 1

        # Create the edges.
        edges = self.get_edges(nodes, edges_file)

        # Construct an unweighted, directed graph from the nodes and edges.
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges) # automatically adds the nodes
        self.tsp_path = self.directed_tsp()
    
    def get_edges(self, nodes, edges_file) -> [(Node,Node)]:
        # Note: this function is specifically designed to work for num_tilt = 3, num_directions = 4.
        all_possible_edges = itertools.permutations(nodes, 2)
        df = pd.read_csv(edges_file)

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

    def directed_tsp(self):
        # Compute shortest path lengths between all pairs of nodes.
        all_pairs_shortest_path_length = dict(nx.floyd_warshall(self.graph))

        # Create a new directed graph which has edges between all pairs of nodes
        # where the weight of the edge is the shortest path length between those 2 nodes.
        H = nx.DiGraph()
        for u in self.graph.nodes():
            for v in self.graph.nodes():
                if u != v:
                    H.add_edge(u, v, weight=all_pairs_shortest_path_length[u][v])

        # Use the NetworkX approximation function for TSP on the directed graph
        tsp_path = traveling_salesman_problem(H, cycle=False, method=greedy_tsp)
        return tsp_path

    def show_graph(self) -> None:
        # Show the graph representation of the robot lab.
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

    @lru_cache
    def shortest_path(self, source: Node, target:Node) -> [Node]:
        # Returns the shortest path between the source and target nodes.
        return nx.shortest_path(self.graph, source=source, target=target)

    @lru_cache
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
        
        # Select the image that is the closest.
        shortest_paths = [self.shortest_path(source, node) for node in nodes]
        return min(shortest_paths, key=len)

    def occlusion_sort(self, obj) -> tuple:
        # Custom sorting function to determine the better view.
        # Better view means lower level of occlusion and larger bounding box.
        occlusion_levels = {'none': 0, 'partial': 1, 'severe': 2}
        object_area = (obj.x2 - obj.x1) * (obj.y2 - obj.y1)
        return (occlusion_levels[obj.occlusion], -object_area)

    @lru_cache
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
        
        # Return the shortest path to that node.
        shortest_path = self.shortest_path(source, the_node)
        return shortest_path
 
    def are_connected(self, source: Node, target: Node) -> bool:
        # Returns whether the target is accessible from the source node.
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

    def get_starting_position(self, num_actions: int, target: Node):
        # Returns a node that is 'num_actions' removed from the target node.
        all_shortest_paths = []
        for node in self.graph.nodes():
            shortest_path = self.shortest_path(node, target)
            if len(shortest_path) - 1 == num_actions:
                return node
        print(f"No node is {num_actions} actions removed from {target}.")

    def random_path(self, source: Node):
        # Randomly navigates until the max_num_actions is reached.
        max_num_actions = 15
        path = [source]
        for i in range(max_num_actions):
            neighbours = list(self.graph.successors(path[-1]))
            path.append(random.choice(neighbours))
        return path

    def forward_only_path(self, source: Node):
        # Only navigate forward until it is no longer possible
        # or the max_num_actions is reached.
        max_num_actions = 15
        path = [source]
        for i in range(max_num_actions):
            neighbours = list(self.graph.successors(path[-1]))
            for neighbour in neighbours:
                if neighbour.direction == path[-1].direction and neighbour.tilt == path[-1].tilt:
                    # Go forward
                    path.append(neighbour)
                    break
        return path

    def shortest_path_conjunction(self, source, list1, list2, optimal_view):
        # Special function to find the shortest path for a conjunction question.
        # For all combinations of the two object instances' occurences, the count
        # question navigation is used (see shortest_path_visiting_nodes).
        pairs = list(itertools.product(list1, list2))

        shortest_path = self.tsp_path
        for pair in pairs:
            path = self.shortest_path_visiting_nodes(source, pair, optimal_view)
            if len(path) < len(shortest_path):
                shortest_path = path

        return shortest_path

    def shortest_path_visiting_nodes(self, source: Node, targets: [Node], optimal_view: bool):
        # This method tries every permutations 
        if any(isinstance(element, list) for element in targets):
            if len(targets) != 2:
                raise ValueError("Num of targets has to be two.")
            return self.shortest_path_conjunction(source, targets[0], targets[1], optimal_view)

        # All possible orders in which the targets can be visited.
        permutations = list(itertools.permutations(targets))
        
        shortest_path = self.tsp_path
        for permutation in permutations:
            path = [source]
            for obj in permutation:
                if optimal_view:
                    path += self.shortest_path_to_optimal_view(path[-1], obj)[1:]
                else:
                    path += self.shortest_path_to_object(path[-1], obj)[1:]
            if len(path) < len(shortest_path):
                shortest_path = path

        return shortest_path

def stitch_images(images):
    # Determine the grid size on the basis of the number of images.
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / grid_size)
    num_cols = min(num_images, grid_size)
    
    # Image dimensions are assumed to be of the same size for all images.
    image_width, image_height = images[0].size
    
    # Create a new blank image with the appropriate size.
    canvas_width = num_cols * image_width
    canvas_height = num_rows * image_height
    canvas = Image.new('RGB', (canvas_width, canvas_height))
    
    # Stitch the images together on the canvas.
    for i, img in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        x = col * image_width
        y = row * image_height
        canvas.paste(img, (x, y))
    
    return canvas

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotations_file",
                        help="The finished annotations file.",
                        required=True,
                        type=str)
    parser.add_argument("-q", "--questions_file",
                        help="File with all the questions",
                        required=True,
                        type=str)
    parser.add_argument("-e", "--edges_file",
                        help="File with the edges between the viewpoints.",
                        required=True,
                        type=str)
    parser.add_argument("-i", "--image_directory",
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

    # Ensure the annotations, questions and edge files are csv files that exist.
    annotations_file = Path(args.annotations_file)
    questions_file = Path(args.questions_file)
    edges_file = Path(args.edges_file)
    for file in (annotations_file, questions_file, edges_file):
        if not (file.exists() and file.is_file() and file.suffix.lower() == ".csv"):
            raise FileNotFoundError(
                f"The file '{file}' does not exist or is not a CSV file."
                )

    # Ensure the input directory already exists.
    image_directory = Path(args.image_directory)
    if not (image_directory.exists() and image_directory.is_dir()):
        raise FileNotFoundError(
            f"The directory '{image_directory}' does not exist."
            )

    # Select only image files from the input directory.
    image_extensions = {".jpg", ".jpeg", ".png"}
    image_files = [
        Path(file) for file in image_directory.iterdir()
        if file.suffix.lower() in image_extensions
    ]

    # Read in the necessary data.
    df_questions = pd.read_csv(questions_file)
    df_questions["object_ids"] = df_questions["object_ids"].apply(eval)

    # Get paths for each question.
    env = Environment(image_files, annotations_file, edges_file)
    
    for _, question in df_questions.iterrows():

        # Get source and target nodes.
        source = random.choice(list(env.graph.nodes()))
        targets = question.object_ids

        if question.question_type.startswith("disjunction"):
            targets = [number for sublist in targets for number in sublist]

        # Find random path and forward only path
        random_path = env.random_path(source)
        forward_only_path = env.forward_only_path(source)

        # Check if the question is multi-target.
        # Disjunction is mono-target because only one object needs to be seen
        # to be able to answer the question.
        if question.question_type.startswith("conjunction") or question.question_type.startswith("count"):
            if question.answer == "yes":
                shortest_path_to_object = env.shortest_path_visiting_nodes(source, targets, optimal_view=False)
                shortest_path_to_optimal_view = env.shortest_path_visiting_nodes(source, targets, optimal_view=True)
            else:
                shortest_path_to_object = env.tsp_path
                shortest_path_to_optimal_view = env.tsp_path
        else:
            # Select the path to the target that is closest.
            shortest_path_to_object = env.tsp_path
            shortest_path_to_optimal_view = env.tsp_path

            for target in targets:
                path_to_object = env.shortest_path_to_object(source, target)
                path_to_optimal_view = env.shortest_path_to_optimal_view(source, target)

                if len(path_to_object) < len(shortest_path_to_object):
                    shortest_path_to_object = path_to_object

                if len(path_to_optimal_view) < len(shortest_path_to_optimal_view):
                    shortest_path_to_optimal_view = path_to_optimal_view

        # Get images from paths
        random_path_images = stitch_images([Image.open(node.image_file) for node in random_path])
        forward_only_path_images = stitch_images([Image.open(node.image_file) for node in forward_only_path])
        shortest_path_to_object_images = stitch_images([Image.open(node.image_file) for node in shortest_path_to_object])
        shortest_path_to_optimal_view_images = stitch_images([Image.open(node.image_file) for node in shortest_path_to_optimal_view])
        
        # Save images.
        random_path_images.save(f"path_images/{question.uid}_random.jpg")
        forward_only_path_images.save(f"path_images/{question.uid}_forward.jpg")
        shortest_path_to_object_images.save(f"path_images/{question.uid}_object.jpg")
        shortest_path_to_optimal_view_images.save(f"path_images/{question.uid}_view.jpg")

        # Save navigation results.
        nav_eval = {
            "question_id": question.uid,
            "source": source,
            "random_path": [random_path],
            "random_path_length": len(random_path),
            "forward_only_path": [forward_only_path],
            "forward_only_path_length": len(forward_only_path),
            "shortest_path_to_object": [shortest_path_to_object],
            "shortest_path_to_object_length": len(shortest_path_to_object),
            "shortest_path_to_optimal_view": [shortest_path_to_optimal_view],
            "shortest_path_to_optimal_view_length": len(shortest_path_to_optimal_view),
        }

        nav_df = pd.DataFrame(nav_eval)
        file_exists = os.path.isfile(args.output_file)
        nav_df.to_csv(args.output_file, mode='a', index=False, header=not file_exists)


if __name__ == "__main__":
    main()
