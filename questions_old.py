def generate_room_questions(rooms_df):
    questions = []
    colour = ""
    location = "house"

    for _, room in rooms_df.iterrows():

        # Is there {colour}{entity} in the {location}?
        existence_question = Question(
            question_type="existence",
            answer="yes" if room._count else "no",
            room_ids=[room._id] if room._count else [],
            colour=colour,
            entity=p.a(room.__name),
            location=location,
        )
        # How many {colour}{entity} are there in the {location}?
        count_question = Question(
            question_type="count",
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
            question_type="conjunction",
            answer="yes" if exists(room1) and exists(room2) else "no",
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
            question_type="disjunction",
            answer="yes" if exists(room1) or exists(room2) else "no",
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
            question_type="conjunction",
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
            question_type="disjunction",
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
            question_type="existence",
            object_ids=objects,
            answer="yes" if len(objects) else "no",
            colour=colour,
            entity=p.a(row.label),
            location=location,
        )
        questions.append(existence_question)

        # How many {colour}{entity} are there in the {location}?
        count_question = Question(
            question_type="count",
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
            question_type="location",
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
            question_type="colour",
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
                    question_type="spatial",
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
                question_type="conjunction",
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
                question_type="disjunction",
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
                question_type="existence",
                object_ids=objects,
                answer="yes" if len(objects) else "no",
                colour=colour,
                entity=p.a(row.label),
                location=location,
            )
            questions.append(existence_question)

            # How many {colour}{entity} are there in the {location}?
            count_question = Question(
                question_type="count",
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
                question_type="colour",
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
                        question_type="spatial",
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
                question_type="conjunction",
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
                question_type="disjunction",
                object_ids=[objects1, objects2],
                answer="yes" if len(objects1) or len(objects2) else "no",
                colour1=p.a(colour1.__name) + " ",
                colour2=p.a(colour2.__name) + " ",
                entity1=obj1.label,
                entity2=obj2.label,
                location=location,
            )
            questions.append(disjunction_question)

    return questions

    for _, colour in colours_df.iterrows():

        # Existence, Count, Location, Spatial.
        for _, row in classes_df.iterrows():
            
            # Select all objects with the current row's label and colour.

            objects = objects_df.loc[(objects_df.label == row.label) & (objects_df.colour.apply(lambda x: colour.__name in x)), "object_id"].tolist()
            
            # Is there {colour}{entity} in the {location}?
            existence_question = Question(
                question_type="existence",
                object_ids=objects,
                answer="yes" if len(objects) else "no",
                colour=p.a(colour.__name) + " ",
                entity=row.label,
                location=location,
            )
            questions.append(existence_question)

            # How many {colour}{entity} are there in the {location}?
            count_question = Question(
                question_type="count",
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
                question_type="location",
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
                        question_type="spatial",
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

                # Is there {colour1}{entity1} and {colour1}{entity2} in the {location}?
                conjunction_question = Question(
                    question_type="conjunction",
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
                    question_type="disjunction",
                    object_ids=[objects1, objects2],
                    answer="yes" if len(objects1) or len(objects2) else "no",
                    colour1=p.a(colour1._name) + " ",
                    colour2=p.a(colour2._name) + " ",
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
                    question_type="existence",
                    object_ids=objects,
                    answer="yes" if len(objects) else "no",
                    colour=p.a(colour.__name) + " ",
                    entity=row.label,
                    location=location,
                )
                questions.append(existence_question)

                # How many {colour}{entity} are there in the {location}?
                count_question = Question(
                    question_type="count",
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
                            question_type="spatial",
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
