"""
Author: Frieso Turkstra
Date: 2024-06-17

This program generates the questions from an environment.
It generates existence, count, colour, location, preposition, and logical
questions on both the house and room level.
The questions alongside additional attributes such as the answer and the 
identifiers of the objects referred to in the question are saved in a csv file.
"""

from pathlib import Path
import pandas as pd
import itertools
import argparse
import inflect
import uuid

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

    @staticmethod
    def get_question(question_type, kwargs):
        df = pd.read_csv("resources/question_templates.csv")
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

def generate_questions(input_file):
    rooms_df = pd.read_csv("resources/rooms.csv")
    labels_df = pd.read_csv("resources/classes.csv")
    objects_df = pd.read_csv(input_file)
    objects_df = objects_df.drop_duplicates("object_id")

    prepositions = ["on", "above", "below", "next_to"]
    for column in ["colour"] + prepositions:
        objects_df[column] = objects_df[column].str.replace("'", '"').apply(eval)

    # Basic questions on house-level
    existence_questions = get_existence_questions(rooms_df, labels_df, objects_df)
    count_questions = get_count_questions(rooms_df, labels_df, objects_df)
    colour_questions = get_colour_questions(objects_df)
    location_questions = get_location_questions(objects_df)
    preposition_questions = get_preposition_questions(objects_df, prepositions)
    logical_questions = get_logical_questions(rooms_df, labels_df, objects_df)

    # Basic questions on room-level
    existence_room_questions = get_existence_room_questions(rooms_df, labels_df, objects_df)
    count_room_questions = get_count_room_questions(rooms_df, labels_df, objects_df)
    colour_room_questions = get_colour_room_questions(rooms_df, objects_df)
    preposition_room_questions = get_preposition_room_questions(rooms_df, objects_df, prepositions)
    logical_room_questions = get_logical_room_questions(rooms_df, labels_df, objects_df)

    # # Complex question (basic questions + room + colour)
    # existence_room_colour_questions = get_existence_room_colour_questions(rooms_df, labels_df, objects_df, colours_df)
    # count_room_colour_questions = get_count_room_colour_questions()
    # preposition_room_colour_questions = get_preposition_room_colour_questions()
    # logical_room_colour_questions = get_logical_room_colour_questions()

    return pd.concat([
        existence_questions,
        count_questions,
        colour_questions,
        location_questions,
        preposition_questions,
        logical_questions,
        existence_room_questions,
        count_room_questions,
        colour_room_questions,
        preposition_room_questions,
        logical_room_questions,
    ], ignore_index=False)

def get_existence_questions(rooms_df, labels_df, objects_df):
    # Is there {entity} in the house?
    room_existence_questions = rooms_df.apply(lambda room: Question(
        question_type="existence",
        answer="yes" if room._count else "no",
        room_ids=[room._id],
        entity=p.a(room.__name),
    ), axis=1)

    object_existence_questions = labels_df.apply(lambda obj: Question(
        question_type="existence",
        answer="yes" if obj.type_count else "no",
        object_ids=objects_df.loc[objects_df.label == obj.label, "object_id"].tolist(),
        entity=p.a(obj.label),
    ), axis=1)

    return pd.concat([room_existence_questions, object_existence_questions], ignore_index=True)

def get_count_questions(rooms_df, labels_df, objects_df):
    # How many {entity} are there in the house?
    room_count_questions = rooms_df.apply(lambda room: Question(
        question_type="count",
        answer=room._count,
        room_ids=[room._id],
        entity=p.plural(room.__name),
    ), axis=1)

    object_count_questions = labels_df.apply(lambda obj: Question(
        question_type = "count",
        answer=obj.type_count,
        object_ids=objects_df.loc[objects_df.label == obj.label, "object_id"].tolist(),
        entity=p.plural(obj.label)
    ), axis=1)

    return pd.concat([room_count_questions, object_count_questions], ignore_index=True)

def get_colour_questions(objects_df):
    # What colour is the {obj}?
    return objects_df.drop_duplicates("label").apply(lambda obj: Question(
        question_type="colour",
        object_ids=(object_ids := objects_df.loc[objects_df.label == obj.label, "object_id"].tolist()),
        answer={obj_id: objects_df.loc[objects_df.object_id == obj_id, "colour"].squeeze() for obj_id in object_ids},
        ambiguous=len(object_ids) > 1,
        obj=obj.label
    ), axis=1)

def get_location_questions(objects_df):
    # What room is the {obj} located in?
    return objects_df.drop_duplicates("label").apply(lambda obj: Question(
        question_type="location",
        object_ids=(object_ids := objects_df.loc[objects_df.label == obj.label, "object_id"].tolist()),
        answer={obj_id: objects_df.loc[objects_df.object_id == obj_id, "location"].squeeze() for obj_id in object_ids},
        ambiguous=len(object_ids) > 1,
        obj=obj.label
    ), axis=1)

def get_preposition_questions(objects_df, prepositions):
    # What is {preposition} the {obj}?
    questions = pd.concat([objects_df.drop_duplicates("label").apply(lambda obj: Question(
        question_type="preposition",
        object_ids=(object_ids := objects_df.loc[(objects_df.label == obj.label) & (objects_df[preposition].str.len() > 0), "object_id"].tolist()),
        answer={obj_id: objects_df.loc[objects_df.object_id == obj_id, preposition].squeeze() for obj_id in object_ids},
        ambiguous=len(object_ids) > 1,
        preposition=preposition,
        obj=obj.label
    ), axis=1) for preposition in prepositions], ignore_index=True)
    return questions[questions.apply(lambda q: bool(q.answer))]

def get_logical_questions(rooms_df, labels_df, objects_df):
    # Is there {entity1} and {entity2} in the house?
    room_logical_questions = pd.Series([Question(
            question_type="logical",
            answer="yes" if room1._count and room2._count else "no",
            room_ids=[room1._id, room2._id],
            entity1=p.a(room1.__name),
            entity2=p.a(room2.__name),
        ) for (_, room1), (_, room2) in itertools.combinations(rooms_df.iterrows(), 2)
    ])

    object_logical_questions = pd.Series([Question(
            question_type="logical",
            answer="yes" if obj1.type_count and obj2.type_count else "no",
            object_ids=[
                objects_df.loc[objects_df.label == obj1.label, "object_id"].tolist(),
                objects_df.loc[objects_df.label == obj2.label, "object_id"].tolist(),
                ],
            entity1=p.a(obj1.label),
            entity2=p.a(obj2.label),
        ) for (_, obj1), (_, obj2) in itertools.combinations(labels_df.iterrows(), 2)
    ])
    return pd.concat([room_logical_questions, object_logical_questions], ignore_index=True)

def get_existence_room_questions(rooms_df, labels_df, objects_df):
    # Is there {obj} in the {room}?
    return pd.concat([labels_df.apply(lambda obj: Question(
        question_type="existence_room",
        object_ids=(object_ids := objects_df.loc[(objects_df.label == obj.label) & (objects_df.location == room.__name), "object_id"].tolist()),
        answer="yes" if len(object_ids) else "no",
        obj=p.a(obj.label),
        room=room.__name,
        room_ids=[room._id]
    ), axis=1) for _, room in rooms_df.loc[rooms_df._count > 0].iterrows()], ignore_index=True)

def get_count_room_questions(rooms_df, labels_df, objects_df):
    # How many {obj} are there in the {room}?
    return pd.concat([labels_df.apply(lambda obj: Question(
        question_type="count_room",
        object_ids=(object_ids := objects_df.loc[(objects_df.label == obj.label) & (objects_df.location == room.__name), "object_id"].tolist()),
        answer=len(object_ids),
        obj=p.plural(obj.label),
        room=room.__name,
        room_ids=[room._id]
    ), axis=1) for _, room in rooms_df.loc[rooms_df._count > 0].iterrows()], ignore_index=True)

def get_colour_room_questions(rooms_df, objects_df):
    # What colour is the {obj} in the {room}?
    questions = pd.concat([objects_df.drop_duplicates("label").apply(lambda obj: Question(
        question_type="colour_room",
        object_ids=(object_ids := objects_df.loc[(objects_df.label == obj.label) & (objects_df.location == room.__name), "object_id"].tolist()),
        answer={obj_id: objects_df.loc[objects_df.object_id == obj_id, "colour"].squeeze() for obj_id in object_ids},
        ambiguous=len(object_ids) > 1,
        obj=obj.label,
        room=room.__name,
        room_ids=[room._id]
    ), axis=1) for _, room in rooms_df.loc[rooms_df._count > 0].iterrows()], ignore_index=True)
    return questions[questions.apply(lambda q: bool(q.answer))]

def get_preposition_room_questions(rooms_df, objects_df, prepositions):
    # What is {preposition} the {obj} in the {room}?
    questions = pd.concat([objects_df.drop_duplicates("label").apply(lambda obj: Question(
        question_type="preposition_room",
        object_ids=(object_ids := objects_df.loc[(objects_df.label == obj.label) & (objects_df.location == room.__name) & (objects_df[preposition].str.len() > 0), "object_id"].tolist()),
        answer={obj_id: objects_df.loc[objects_df.object_id == obj_id, preposition].squeeze() for obj_id in object_ids},
        ambiguous=len(object_ids) > 1,
        preposition=preposition,
        obj=obj.label,
        room=room.__name,
        room_ids=[room._id]
    ), axis=1) for preposition in prepositions for _, room in rooms_df.loc[rooms_df._count > 0].iterrows()], ignore_index=True)
    return questions[questions.apply(lambda q: bool(q.answer))]

def get_logical_room_questions(rooms_df, labels_df, objects_df):
    # Is there {obj1} and {obj2} in the {room}?
    return pd.Series([Question(
            question_type="logical_room",
            object_ids=(object_ids := [
                objects_df.loc[(objects_df.label == obj1.label) & (objects_df.location == room.__name), "object_id"].tolist(),
                objects_df.loc[(objects_df.label == obj2.label) & (objects_df.location == room.__name), "object_id"].tolist(),
                ]),
            answer="yes" if len(object_ids[0]) and len(object_ids[1]) else "no",
            room_ids=[room._id],
            obj1=p.a(obj1.label),
            obj2=p.a(obj2.label),
            room=room.__name
        ) for _, room in rooms_df.loc[rooms_df._count > 0].iterrows() for (_, obj1), (_, obj2) in itertools.combinations(labels_df.iterrows(), 2)])

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
    questions = pd.DataFrame([vars(question) for question in generate_questions(input_file)])
    questions.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    main()
