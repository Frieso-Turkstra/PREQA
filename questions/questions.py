"""
Author: Frieso Turkstra
Date: 2024-06-17

This program generates the questions from an environment.
It generates existence, count, colour, location, spatial, conjunction and 
disjunction questions.
The questions alongside additional attributes such as the answer and the 
identifiers of the objects referred to in the question are saved in a csv file.
"""

from nltk.corpus import wordnet as wn
from pathlib import Path
import pandas as pd
import itertools
import argparse
import inflect
import uuid
import os

# Used to get the correct plural form or indefinite article for a noun.
p = inflect.engine()


class Question:
    def __init__(self, question_type, answer, ambiguous=False, object_ids=None, room_ids=None, **kwargs):
        # The kwargs are used to pass in the values for the template slots.
        # The name and number of these may vary.
        self.question = Question.get_question(question_type, kwargs)
        self.question_type = question_type
        self.answer = answer
        self.ambiguous = ambiguous
        self.object_ids = [] if object_ids is None else object_ids
        self.room_ids = [] if room_ids is None else room_ids
        self.uid = str(uuid.uuid4())
        self.origin = "template"
        self.kwargs = kwargs

    @staticmethod
    def get_question(question_type, kwargs):
        global resources_directory
        # Finds the right template and fills it in with the kwargs.
        df = pd.read_csv(f"{resources_directory}/templates.csv")
        question_type = question_type.split("_")[0]
        template = df.loc[df.question_type == question_type, "template"].squeeze()
        kwargs = {k: v.replace("_", " ") for k, v in kwargs.items()}
        return template.format(**kwargs)
    

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file",
                        help="File with the annotated images and objects.",
                        required=True,
                        type=str)
    parser.add_argument("-r", "--resources_directory",
                        help="Directory with the question templates, object classes, rooms and colours.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_file",
                        help="File to which the questions are saved.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args


def get_hypernyms(synset, max_steps):
    # Get all hypernyms of a synset up to 'max_steps' steps removed.
    hypernyms = set()
    current_level = {synset}
    for _ in range(max_steps):
        next_level = set()
        for syn in current_level:
            hypernyms.update(syn.hypernyms())
            next_level.update(syn.hypernyms())
        current_level = next_level
    return hypernyms

def common_ancestor_within_steps(wordnet_sense1, wordnet_sense2, max_steps):
    # Determine if two WordNet synsets have a common ancestor within 'max_steps'
    synset1 = wn.synset(wordnet_sense1)
    synset2 = wn.synset(wordnet_sense2)
    hypernyms1 = get_hypernyms(synset1, max_steps)
    hypernyms2 = get_hypernyms(synset2, max_steps)
    
    # Check for common hypernyms
    common_hypernyms = hypernyms1.intersection(hypernyms2)
    return len(common_hypernyms) > 0

def generate_questions(input_file, resources_directory, output_file):
    # Read in and preprocess data.
    rooms_df = pd.read_csv(f"{resources_directory}/rooms.csv")
    classes_df = pd.read_csv(f"{resources_directory}/classes.csv")
    colours_df = pd.read_csvf(f"{resources_directory}/colours.csv")
    objects_df = pd.read_csv(input_file)
    objects_df = objects_df.drop_duplicates("object_id")

    prepositions = ["on", "above", "below", "next_to"]
    for column in ["colour"] + prepositions:
        objects_df[column] = objects_df[column].str.replace("'", '"').apply(eval)

    # Generate questions
    questions = generate_basic_questions(classes_df, objects_df)
    questions = generate_location_questions(rooms_df, classes_df, objects_df)
    questions = generate_colour_questions(colours_df, classes_df, objects_df)
    questions = generate_colour_location_questions(rooms_df, colours_df, classes_df, objects_df)

    # Save questions
    df = pd.DataFrame([vars(q) for q in questions])
    file_exists = os.path.isfile(output_file)
    df.to_csv(output_file, mode='a', index=False, header=not file_exists)


def generate_basic_questions(classes_df, objects_df):
    questions = []
    colour = ""
    location = "house"

    # Conjunction and Disjunction.
    for (_, obj1), (_, obj2) in itertools.combinations(classes_df.iterrows(), 2):
        
        # Only compare objects that have a common ancestor within two steps.
        max_steps = 2
        if not common_ancestor_within_steps(obj1.wordnet_sense, obj2.wordnet_sense, max_steps):
            continue

        objects1 = objects_df.loc[objects_df.label == obj1.label, "object_id"].tolist()
        objects2 = objects_df.loc[objects_df.label == obj2.label, "object_id"].tolist()

        # Is there {colour1}{entity1} and {colour1}{entity2} in the {location}?
        conjunction_question = Question(
            question_type="conjunction_object",
            object_ids=[objects1, objects2],
            answer="yes" if len(objects1) and len(objects2) else "no",
            colour1=colour,
            colour2=colour,
            entity1=p.a(obj1.label), 
            entity2=p.a(obj2.label),
            location=location,
        )
        questions.append(conjunction_question)

        # Is there {colour2}{entity1} or {colour2}{entity2} in the {location}?
        disjunction_question = Question(
            question_type="disjunction_object",
            object_ids=[objects1, objects2],
            answer="yes" if len(objects1) or len(objects2) else "no",
            colour1=colour,
            colour2=colour,
            entity1=p.a(obj1.label),
            entity2=p.a(obj2.label),
            location=location,
        )
        questions.append(disjunction_question)
    
    # Existence, Count, Location, Colour, Spatial.
    for _, row in classes_df.iterrows():
        
        # Select all objects with the current row's label.
        objects = objects_df.loc[objects_df.label == row.label, "object_id"].tolist()

        # Is there {colour}{entity} in the {location}?
        existence_question = Question(
            question_type="existence_object",
            object_ids=objects,
            answer="yes" if len(objects) else "no",
            colour=colour,
            entity=p.a(row.label),
            location=location,
        )
        questions.append(existence_question)

        # How many {colour}{entity} are there in the {location}?
        count_question = Question(
            question_type="count_object",
            object_ids=objects,
            answer=len(objects),
            colour=colour,
            entity=p.plural(row.label),
            location=location,
        )
        questions.append(count_question)
        
        # Location, colour and spatial questions assume object existence.
        if not objects:
            continue

        # What room is the {colour}{obj} located in? 
        location_question = Question(
            question_type="location_object",
            object_ids=objects,
            answer={
                obj_id: objects_df.loc[objects_df.object_id == obj_id, "location"].squeeze()
                for obj_id in objects
                },
            ambiguous=len(objects) > 1,
            colour=colour,
            obj=row.label,
        )
        questions.append(location_question)

        # What colour is the {obj}{location}?
        colour_question = Question(
            question_type="colour_object",
            object_ids=objects,
            answer={
                obj_id: objects_df.loc[objects_df.object_id == obj_id, "colour"].squeeze()
                for obj_id in objects
                },
            ambiguous=len(objects) > 1,
            obj=row.label,
            location="",
        )
        questions.append(colour_question)

        # What is {preposition} the {colour}{obj}{location}?
        for preposition in ("on", "above", "below", "next_to"):

            objects_with_preposition = []
            for obj in objects:
                if not objects_df.loc[(objects_df.object_id == obj) & (objects_df[preposition].str.len() > 0)].empty:
                    objects_with_preposition.append(obj)

            if objects_with_preposition:
                spatial_question = Question(
                    question_type="spatial_object",
                    object_ids=objects_with_preposition,
                    answer={
                        obj: objects_df.loc[objects_df.object_id == obj, preposition].squeeze()
                        for obj in objects_with_preposition
                        },
                    ambiguous=len(objects_with_preposition) > 1,
                    preposition=preposition,
                    colour=colour,
                    obj=row.label,
                    location="",
                )
                questions.append(spatial_question)

    return questions

def generate_location_questions(rooms_df, classes_df, objects_df):
    questions = []
    colour = ""

    for _, room in rooms_df.iterrows():
        
        # Assumes room existence.
        if not room._count:
            continue

        location = room.__name

        # Conjunction and Disjunction.
        for (_, obj1), (_, obj2) in itertools.combinations(classes_df.iterrows(), 2):
            
            # Only compare objects that have a common ancestor within two steps.
            max_steps = 2
            if not common_ancestor_within_steps(obj1.wordnet_sense, obj2.wordnet_sense, max_steps):
                continue

            objects1 = objects_df.loc[(objects_df.label == obj1.label) & (objects_df.location == room.__name), "object_id"].tolist()
            objects2 = objects_df.loc[(objects_df.label == obj2.label) & (objects_df.location == room.__name), "object_id"].tolist()

            # Is there {colour1}{entity1} and {colour1}{entity2} in the {location}?
            conjunction_question = Question(
                question_type="conjunction_location",
                object_ids=[objects1, objects2],
                answer="yes" if len(objects1) and len(objects2) else "no",
                colour1=colour,
                colour2=colour,
                entity1=p.a(obj1.label), 
                entity2=p.a(obj2.label),
                location=location,
            )
            questions.append(conjunction_question)

            # Is there {colour2}{entity1} or {colour2}{entity2} in the {location}?
            disjunction_question = Question(
                question_type="disjunction_location",
                object_ids=[objects1, objects2],
                answer="yes" if len(objects1) or len(objects2) else "no",
                colour1=colour,
                colour2=colour,
                entity1=p.a(obj1.label),
                entity2=p.a(obj2.label),
                location=location,
            )
            questions.append(disjunction_question)

        # Existence, Count, Colour, Spatial.
        for _, row in classes_df.iterrows():
            
            # Select all objects with the current row's label and the room.
            objects = objects_df.loc[(objects_df.label == row.label) & (objects_df.location == room.__name), "object_id"].tolist()

            # Is there {colour}{entity} in the {location}?
            existence_question = Question(
                question_type="existence_location",
                object_ids=objects,
                answer="yes" if len(objects) else "no",
                colour=colour,
                entity=p.a(row.label),
                location=location,
            )
            questions.append(existence_question)

            # How many {colour}{entity} are there in the {location}?
            count_question = Question(
                question_type="count_location",
                object_ids=objects,
                answer=len(objects),
                colour=colour,
                entity=p.plural(row.label),
                location=location,
            )
            questions.append(count_question)
            
            # Colour and spatial questions assume object existence.
            if not objects:
                continue

            # What colour is the {obj}{location}?
            colour_question = Question(
                question_type="colour_location",
                object_ids=objects,
                answer={
                    obj_id: objects_df.loc[objects_df.object_id == obj_id, "colour"].squeeze()
                    for obj_id in objects
                    },
                ambiguous=len(objects) > 1,
                obj=row.label,
                location=f" in the {location}",
            )
            questions.append(colour_question)

            # What is {preposition} the {colour}{obj}{location}?
            for preposition in ("on", "above", "below", "next_to"):

                objects_with_preposition = []
                for obj in objects:
                    if not objects_df.loc[(objects_df.object_id == obj) & (objects_df[preposition].str.len() > 0)].empty:
                        objects_with_preposition.append(obj)

                if objects_with_preposition:
                    spatial_question = Question(
                        question_type="spatial_location",
                        object_ids=objects_with_preposition,
                        answer={
                            obj: objects_df.loc[objects_df.object_id == obj, preposition].squeeze()
                            for obj in objects_with_preposition
                            },
                        ambiguous=len(objects_with_preposition) > 1,
                        preposition=preposition,
                        colour=colour,
                        obj=row.label,
                        location=f" in the {location}",
                    )
                    questions.append(spatial_question)

    return questions
    
def generate_colour_questions(colours_df, classes_df, objects_df):
    questions = []
    location = "house"

    # Conjunction and Disjunction.
    for (_, obj1), (_, obj2) in itertools.combinations(classes_df.iterrows(), 2):
        
        # Only compare objects that have a common ancestor within two steps.
        max_steps = 2
        if not common_ancestor_within_steps(obj1.wordnet_sense, obj2.wordnet_sense, max_steps):
            continue

        for (_, colour1), (_, colour2) in itertools.combinations(colours_df.iterrows(), 2):

            objects1 = objects_df.loc[(objects_df.label == obj1.label) & (objects_df.colour.apply(lambda x: colour1.__name in x)), "object_id"].tolist()
            objects2 = objects_df.loc[(objects_df.label == obj2.label) & (objects_df.colour.apply(lambda x: colour2.__name in x)), "object_id"].tolist()

            # Is there {colour1}{entity1} and {colour1}{entity2} in the {location}?
            conjunction_question = Question(
                question_type="conjunction_colour",
                object_ids=[objects1, objects2],
                answer="yes" if len(objects1) and len(objects2) else "no",
                colour1=p.a(colour1.__name) + " ",
                colour2=p.a(colour2.__name) + " ",
                entity1=obj1.label, 
                entity2=obj2.label,
                location=location,
            )
            questions.append(conjunction_question)
                
            # Is there {colour2}{entity1} or {colour2}{entity2} in the {location}?
            disjunction_question = Question(
                question_type="disjunction_colour",
                object_ids=[objects1, objects2],
                answer="yes" if len(objects1) or len(objects2) else "no",
                colour1=p.a(colour1.__name) + " ",
                colour2=p.a(colour2.__name) + " ",
                entity1=obj1.label,
                entity2=obj2.label,
                location=location,
            )
            questions.append(disjunction_question)

    for _, colour in colours_df.iterrows():

        # Existence, Count, Location, Spatial.
        for _, row in classes_df.iterrows():
            
            # Select all objects with the current row's label and colour.

            objects = objects_df.loc[(objects_df.label == row.label) & (objects_df.colour.apply(lambda x: colour.__name in x)), "object_id"].tolist()
            
            # Is there {colour}{entity} in the {location}?
            existence_question = Question(
                question_type="existence_colour",
                object_ids=objects,
                answer="yes" if len(objects) else "no",
                colour=p.a(colour.__name) + " ",
                entity=row.label,
                location=location,
            )
            questions.append(existence_question)

            # How many {colour}{entity} are there in the {location}?
            count_question = Question(
                question_type="count_colour",
                object_ids=objects,
                answer=len(objects),
                colour=colour.__name + " ",
                entity=p.plural(row.label),
                location=location,
            )
            questions.append(count_question)
            
            # Location, colour and spatial questions assume object existence.
            if not objects:
                continue

            # What room is the {colour}{obj} located in? 
            location_question = Question(
                question_type="location_colour",
                object_ids=objects,
                answer={
                    obj_id: objects_df.loc[objects_df.object_id == obj_id, "location"].squeeze()
                    for obj_id in objects
                    },
                ambiguous=len(objects) > 1,
                colour=colour.__name + " ",
                obj=row.label,
            )
            questions.append(location_question)

            # What is {preposition} the {colour}{obj}{location}?
            for preposition in ("on", "above", "below", "next_to"):

                objects_with_preposition = []
                for obj in objects:
                    if not objects_df.loc[(objects_df.object_id == obj) & (objects_df[preposition].str.len() > 0)].empty:
                        objects_with_preposition.append(obj)

                if objects_with_preposition:
                    spatial_question = Question(
                        question_type="spatial_colour",
                        object_ids=objects_with_preposition,
                        answer={
                            obj: objects_df.loc[objects_df.object_id == obj, preposition].squeeze()
                            for obj in objects_with_preposition
                            },
                        ambiguous=len(objects_with_preposition) > 1,
                        preposition=preposition,
                        colour=colour.__name + " ",
                        obj=row.label,
                        location="",
                    )
                    questions.append(spatial_question)
                    continue

    return questions
    
def generate_colour_location_questions(rooms_df, colours_df, classes_df, objects_df):
    questions = []

    total = 0
    yes_conjunction = 0
    yes_disjunction = 0

    for _, room in rooms_df.iterrows():

        # Assumes room existence.
        if not room._count:
            continue

        location = room.__name

        # Conjunction and Disjunction.
        for (_, obj1), (_, obj2) in itertools.combinations(classes_df.iterrows(), 2):
            
            # Only compare objects that have a common ancestor within two steps.
            max_steps = 2
            if not common_ancestor_within_steps(obj1.wordnet_sense, obj2.wordnet_sense, max_steps):
                continue

            for (_, colour1), (_, colour2) in itertools.combinations(colours_df.iterrows(), 2):

                objects1 = objects_df.loc[(objects_df.label == obj1.label) & (objects_df.location == room.__name) & (objects_df.colour.apply(lambda x: colour1.__name in x)), "object_id"].tolist()
                objects2 = objects_df.loc[(objects_df.label == obj2.label) & (objects_df.location == room.__name) & (objects_df.colour.apply(lambda x: colour2.__name in x)), "object_id"].tolist()

                total += 1
                yes_conjunction += (len(objects1) and len(objects2))
                yes_disjunction += (len(objects1) or len(objects2))

                # Is there {colour1}{entity1} and {colour1}{entity2} in the {location}?
                conjunction_question = Question(
                    question_type="conjunction_colour_location",
                    object_ids=[objects1, objects2],
                    answer="yes" if len(objects1) and len(objects2) else "no",
                    colour1=p.a(colour1.__name) + " ",
                    colour2=p.a(colour2.__name) + " ",
                    entity1=obj1.label, 
                    entity2=obj2.label,
                    location=location,
                )
                questions.append(conjunction_question)

                # Is there {colour2}{entity1} or {colour2}{entity2} in the {location}?
                disjunction_question = Question(
                    question_type="disjunction_colour_location",
                    object_ids=[objects1, objects2],
                    answer="yes" if len(objects1) or len(objects2) else "no",
                    colour1=p.a(colour1.__name) + " ",
                    colour2=p.a(colour2.__name) + " ",
                    entity1=obj1.label,
                    entity2=obj2.label,
                    location=location,
                )
                questions.append(disjunction_question)

        for _, colour in colours_df.iterrows():

            # Existence, Count, Colour, Spatial.
            for _, row in classes_df.iterrows():
                
                # Select all objects with the current row's label and the room.
                objects = objects_df.loc[(objects_df.label == row.label) & (objects_df.location == room.__name) & (objects_df.colour.apply(lambda x: colour.__name in x)), "object_id"].tolist()

                # Is there {colour}{entity} in the {location}?
                existence_question = Question(
                    question_type="existence_colour_location",
                    object_ids=objects,
                    answer="yes" if len(objects) else "no",
                    colour=p.a(colour.__name) + " ",
                    entity=row.label,
                    location=location,
                )
                questions.append(existence_question)

                # How many {colour}{entity} are there in the {location}?
                count_question = Question(
                    question_type="count_colour_location",
                    object_ids=objects,
                    answer=len(objects),
                    colour=colour.__name + " ",
                    entity=p.plural(row.label),
                    location=location,
                )
                questions.append(count_question)

                # Spatial questions assume object existence.
                if not objects:
                    continue

                # What is {preposition} the {colour}{obj}{location}?
                for preposition in ("on", "above", "below", "next_to"):

                    objects_with_preposition = []
                    for obj in objects:
                        if not objects_df.loc[(objects_df.object_id == obj) & (objects_df[preposition].str.len() > 0)].empty:
                            objects_with_preposition.append(obj)

                    if objects_with_preposition:
                        spatial_question = Question(
                            question_type="spatial_colour_location",
                            object_ids=objects_with_preposition,
                            answer={
                                obj: objects_df.loc[objects_df.object_id == obj, preposition].squeeze()
                                for obj in objects_with_preposition
                                },
                            ambiguous=len(objects_with_preposition) > 1,
                            preposition=preposition,
                            colour=colour.__name + " ",
                            obj=row.label,
                            location=f" in the {location}",
                        )
                        questions.append(spatial_question)

    return questions


def main():
    global resources_directory

    # Read in the command-line arguments.
    args = create_arg_parser()

    # Ensure the annotations file is a csv file that exists.
    input_file = Path(args.input_file)
    if not (
        input_file.exists() and
        input_file.is_file() and
        input_file.suffix.lower() == ".csv"
    ):
        raise FileNotFoundError(
            f"The file '{input_file}' does not exist or is not a CSV file."
            )

    # Ensure the resources directory already exists.
    resources_directory = Path(args.resources_directory)
    if not (resources_directory.exists() and resources_directory.is_dir()):
        raise FileNotFoundError(
            f"The directory '{resources_directory}' does not exist."
            )

    # Generate and save questions.
    questions = pd.DataFrame([vars(question) for question in generate_questions(input_file, resources_directory, args.output_file)])
    questions.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
