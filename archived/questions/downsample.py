import pandas as pd
import argparse
import os

# Downsample only the existence, count, disjunction and conjunction questions
NEEDS_DOWNSAMPLING = [
    "conjunction_room", "conjunction_object", "conjunction_location", "conjunction_colour", "conjunction_colour_location",
    "disjunction_room", "disjunction_object", "disjunction_location", "disjunction_colour", "disjunction_colour_location",
    "existence_room", "existence_object", "existence_location", "existence_colour", "existence_colour_location",
    "count_room", "count_object", "count_location", "count_colour", "count_colour_location",
    ]


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file",
                        help="File with all the questions.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_file",
                        help="File to which the downsampled questions are saved.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args



def downsample(df, output_file):

    # For each question type variant...
    for question_type in df.question_type.unique():

        # select all rows with that question type variant.
        sub_df = df[df.question_type == question_type]

        if question_type in needs_downsampling:
            
            if question_type.startswith("disjunction"):
                # The yes and no categories for the disjunction questions are
                # downsampled to the number of yesses on conjunction questions.
                yes_df = sub_df[sub_df.answer == "yes"]
                no_df = sub_df[sub_df.answer == "no"]

                minority_class = len(df[(df.question_type == question_type.replace("disjunction", "conjunction")) & (df.answer == "yes")])
                
                yes_downsampled_df = yes_df.sample(n=minority_class, random_state=42)
                no_downsampled_df = no_df.sample(n=minority_class, random_state=42)
                sub_df = pd.concat([yes_downsampled_df, no_downsampled_df])
            else:
                if question_type.startswith("count"):
                    # Questions with the answer zero are downsampled until they
                    # are as frequent as non-zero answers.
                    to_downsample_df = sub_df[sub_df.answer.astype(int) == 0]
                    remain_df = sub_df[sub_df.answer.astype(int) > 0]
                else:
                    # Conjunction questions are downsampled until the no-answers
                    # are as frequent as the yes-answers.
                    to_downsample_df = sub_df[sub_df.answer == "no"]
                    remain_df = sub_df[sub_df.answer == "yes"]

                downsampled_df = to_downsample_df.sample(n=len(remain_df), random_state=42)
                sub_df = pd.concat([remain_df, downsampled_df])

        # Save the questions to the output file.
        file_exists = os.path.isfile(output_file)
        sub_df.to_csv(output_file, mode="a", index=False, header=not file_exists)


def main():
    # Read in the command-line arguments.
    args = create_arg_parser()

    # Ensure the questions file is a csv file that exists.
    input_file = Path(args.input_file)
    if not (
        input_file.exists() and
        input_file.is_file() and
        input_file.suffix.lower() == ".csv"
    ):
        raise FileNotFoundError(
            f"The file '{input_file}' does not exist or is not a CSV file."
            )

    # Read in the data.
    questions_df = pd.read_csv(input_file)
    downsample(questions_df, args.output_file)
    

if __name__ == "__main__":
    main()
