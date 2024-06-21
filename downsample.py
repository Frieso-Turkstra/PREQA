import pandas as pd
import os

df = pd.read_csv("data/questions.csv")

needs_downsampling = [
    "conjunction_room", "conjunction_object", "conjunction_location", "conjunction_colour", "conjunction_colour_location",
    "disjunction_room", "disjunction_object", "disjunction_location", "disjunction_colour", "disjunction_colour_location",
    "existence_room", "existence_object", "existence_location", "existence_colour", "existence_colour_location",
    "count_room", "count_object", "count_location", "count_colour", "count_colour_location",
    ]


for question_type in df.question_type.unique():
    sub_df = df[df.question_type == question_type]

    if question_type in needs_downsampling:
        if question_type.startswith("disjunction"):
            yes_df = sub_df[sub_df.answer == "yes"]
            no_df = sub_df[sub_df.answer == "no"]

            minority_class = len(df[(df.question_type == question_type.replace("disjunction", "conjunction")) & (df.answer == "yes")])
            
            yes_downsampled_df = yes_df.sample(n=minority_class, random_state=42)
            no_downsampled_df = no_df.sample(n=minority_class, random_state=42)
            sub_df = pd.concat([yes_downsampled_df, no_downsampled_df])
        else:
            if question_type.startswith("count"):
                to_downsample_df = sub_df[sub_df.answer.astype(int) == 0]
                remain_df = sub_df[sub_df.answer.astype(int) > 0]
            else:
                to_downsample_df = sub_df[sub_df.answer == "no"]
                remain_df = sub_df[sub_df.answer == "yes"]

            downsampled_df = to_downsample_df.sample(n=len(remain_df), random_state=42)
            sub_df = pd.concat([remain_df, downsampled_df])

    # Save questions
    output_file = "data/questions_downsampled.csv"

    # Check if the file exists
    file_exists = os.path.isfile(output_file)
    sub_df.to_csv(output_file, mode='a', index=False, header=not file_exists)

