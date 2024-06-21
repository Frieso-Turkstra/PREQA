"""
Author: Frieso Turkstra
Date: 2024-06-17

This program generates the questions from an environment.
It generates existence, count, colour, location, preposition, and logical
questions on both the house and room level.
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


p = inflect.engine()


class Question:
    def __init__(self, question_type, answer, ambiguous=False, object_ids=None, room_ids=None, **kwargs):
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
        df = pd.read_csv("resources/question_templates.csv")
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
    parser.add_argument("-o", "--output_file",
                        help="File to which the questions are saved.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def get_hypernyms(synset, max_steps):
    """
    Get all hypernyms of a synset up to max_steps steps removed.
    
    :param synset: WordNet synset
    :param max_steps: Maximum number of steps to trace back hypernyms
    :return: A set of hypernyms up to max_steps
    """
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
    """
    Determine if two WordNet synsets have a common ancestor within max_steps.
    
    :param synset1: First WordNet synset
    :param synset2: Second WordNet synset
    :param max_steps: Maximum number of steps to trace back hypernyms
    :return: True if there is a common ancestor, False otherwise
    """
    synset1 = wn.synset(wordnet_sense1)
    synset2 = wn.synset(wordnet_sense2)
    hypernyms1 = get_hypernyms(synset1, max_steps)
    hypernyms2 = get_hypernyms(synset2, max_steps)
    
    # Check for common hypernyms
    common_hypernyms = hypernyms1.intersection(hypernyms2)
    return len(common_hypernyms) > 0

def generate_questions(input_file):
    # Read in and preprocess data.
    rooms_df = pd.read_csv("resources/rooms.csv")
    classes_df = pd.read_csv("resources/classes.csv")
    colours_df = pd.read_csv("resources/colours.csv")
    objects_df = pd.read_csv(input_file)
    objects_df = objects_df.drop_duplicates("object_id")

    prepositions = ["on", "above", "below", "next_to"]
    for column in ["colour"] + prepositions:
        objects_df[column] = objects_df[column].str.replace("'", '"').apply(eval)

    # Generate questions
    # questions = generate_room_questions(rooms_df)
    # questions = generate_basic_questions(classes_df, objects_df)
    # questions = generate_location_questions(rooms_df, classes_df, objects_df)
    # questions = generate_colour_questions(colours_df, classes_df, objects_df)
    questions = generate_colour_location_questions(rooms_df, colours_df, classes_df, objects_df)

    # Save questions
    df = pd.DataFrame([vars(q) for q in questions])
    output_file = "test.csv"

    # Check if the file exists
    file_exists = os.path.isfile(output_file)
    df.to_csv(output_file, mode='a', index=False, header=not file_exists)

def generate_room_questions(rooms_df):
    questions = []
    colour = ""
    location = "house"

    for _, room in rooms_df.iterrows():

        # Is there {colour}{entity} in the {location}?
        existence_question = Question(
            question_type="existence_room",
            answer="yes" if room._count else "no",
            room_ids=[room._id] if room._count else [],
            colour=colour,
            entity=p.a(room.__name),
            location=location,
        )
        # How many {colour}{entity} are there in the {location}?
        count_question = Question(
            question_type="count_room",
            answer=room._count,
            room_ids=[room._id] if room._count else [],
            colour=colour,
            entity=p.plural(room.__name),
            location=location,
        )

        questions += [existence_question, count_question]

    for (_, room1), (_, room2) in itertools.combinations(rooms_df.iterrows(), 2):

        # Is there {colour1}{entity1} and {colour1}{entity2} in the {location}?
        conjunction_question = Question(
            question_type="conjunction_room",
            answer="yes" if room1._count and room2._count else "no",
            room_ids=[
                [room1._id] if room1._count else [],
                [room2._id] if room2._count else []
                ],
            colour1=colour,
            colour2=colour,
            entity1=p.a(room1.__name),
            entity2=p.a(room2.__name),
            location=location,
        )

        # Is there {colour2}{entity1} or {colour2}{entity2} in the {location}?
        disjunction_question = Question(
            question_type="disjunction_room",
            answer="yes" if room1._count or room2._count else "no",
            room_ids=[
                [room1._id] if room1._count else [],
                [room2._id] if room2._count else []
                ],
            colour1=colour,
            colour2=colour,
            entity1=p.a(room1.__name),
            entity2=p.a(room2.__name),
            location=location,
        )

        questions += [conjunction_question, disjunction_question]

    return questions

def generate_basic_questions(classes_df, objects_df):
    questions = []
    colour = ""
    location = "house"

    # Conjunction and Disjunction.
    for (_, obj1), (_, obj2) in itertools.combinations(classes_df.iterrows(), 2):
        
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

    print("Total:", total)
    print("Conjunction:", yes_conjunction)
    print("Disjunction:", yes_disjunction)
    return questions


def main():
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

    # Generate and save questions.
    generate_questions(input_file)
    # questions = pd.DataFrame([vars(question) for question in generate_questions(input_file)])
    # questions.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
