from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast

df_labels = pd.read_csv("data/annotated.csv")
df_labels = df_labels.drop_duplicates("object_id")


def get_distribution(df, question_type):
    # For conjunction, disjunction, count and existence questions.
    distribution = df.groupby("question_type")["answer"].value_counts(normalize=True).unstack().fillna(0)
    return distribution


def get_location_distribution(df, question_type):
    def extract_location(answer):
        try:
            answer_dict = ast.literal_eval(answer)
            return answer_dict[next(iter(answer_dict))]
        except (ValueError, SyntaxError, KeyError):
            return None

    # Apply the function to extract locations
    df["location"] = df["answer"].apply(extract_location)

    # Group by 'question_type' and 'location' to calculate the distribution
    distribution = df.groupby(["question_type", "location"]).size().unstack().fillna(0)
    distribution = distribution.div(distribution.sum(axis=1), axis=0)

    return distribution

def get_colour_distribution(df_questions, question_type):
    pass


def main():
    question_type = "colour"
    merge_object_room_questions = True
    df_questions = pd.read_csv("data/questions.csv")
    df_questions = df_questions[df_questions.question_type.str.startswith(question_type)]

    if merge_object_room_questions:
        df_questions["question_type"] = df_questions["question_type"].replace({
            f"{question_type}_object": question_type,
            f"{question_type}_room": question_type,
        })

    if question_type in ("existence", "conjunction", "disjunction", "count"):
        distribution = get_distribution(df_questions, question_type)
    elif question_type == "location":
        distribution = get_location_distribution(df_questions, question_type)
    elif question_type == "colour":
        distribution = get_colour_distribution(df_questions, question_type)

    # Prepare data for plotting
    question_types = distribution.index.tolist()
    answers = distribution.columns.tolist()

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set bar width
    bar_width = 0.5

    # Define named colors for the labels
    named_colours = sns.color_palette("hsv", len(answers))
    answer_to_color = dict(zip(answers, named_colours))

    # Create bars
    bars = []
    for i, question_type in enumerate(question_types):
        # Get counts.
        counts = distribution.loc[question_type].tolist()

        # Create bars.
        bottom = 0
        for answer, count in zip(answers, counts):
            bars.append(ax.bar(i, count, bar_width, bottom=bottom, label=answer if i == 0 else "", color=answer_to_color[answer]))
            bottom += count  
            print(bottom)

    # Set x-axis labels.
    ax.set_xlabel("Question Types")
    ax.set_ylabel("Proportion of Answers (%)")
    ax.set_title("Proportion of 'Yes' and 'No' Answers by Question Type")

    # Set X-axis ticks and labels.
    ax.set_xticks(range(len(question_types)))
    ax.set_xticklabels(question_types)

    # Set Y-axis range.
    ax.set_ylim(0, 1)

    # Add legend
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
#-----COLOUR--------------------------------------------------------------------

# Function to extract colours from the answer string
# def extract_colours(answer):
#     try:
#         answer_dict = ast.literal_eval(answer)
#         colours = []
#         for colour_list in answer_dict.values():
#             colours.extend(colour_list)
#         return colours
#     except (ValueError, SyntaxError, KeyError):
#         return []

# # Apply the function to extract colors
# df["colour"] = df["answer"].apply(extract_colours)

# # Normalize the DataFrame to handle multiple colors per answer
# df_normalized = df.explode("colour").dropna(subset=["colour"])

# # Group by 'question_type' and 'colors' to calculate the distribution
# distribution = df_normalized.groupby(["question_type", "colour"]).size().unstack().fillna(0)
# print(distribution)


#-----SPATIAL-------------------------------------------------------------------
# def extract_numbers(answer):
#     try:
#         answer_dict = ast.literal_eval(answer)
#         numbers = []
#         for number_list in answer_dict.values():
#             numbers.extend(number_list)
#         return numbers
#     except (ValueError, SyntaxError, KeyError):
#         return []

# # Apply the function to extract numbers
# df["numbers"] = df["answer"].apply(extract_numbers)

# # Normalize the DataFrame to handle multiple numbers per answer
# df_normalized = df.explode("numbers").dropna(subset=["numbers"])

# # Ensure the numbers column is of type int for merging
# df_normalized["numbers"] = df_normalized["numbers"].astype(int)

# # Merge with the annotated data to map numbers to labels
# df_merged = df_normalized.merge(df_labels, left_on="numbers", right_on="object_id", how="left")

# # Group by 'question_type' and 'label' and count the occurrences
# distribution = df_merged.groupby(["question_type", "label"]).size().unstack().fillna(0)

# # Display the distribution
# print(print(distribution))

# # Select all question types of spatial_colour.
# # Check all apple questions. should be 8

# print(df.loc[df["question_type"] == "spatial_colour", "answer"])
