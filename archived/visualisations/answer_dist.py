from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ast


def get_distribution(df):
    # For conjunction, disjunction, count, and existence questions.
    distribution = df.groupby("main_question_type")["answer"].value_counts(normalize=True).unstack().fillna(0)
    return distribution

def plot_combined_distribution(distribution_downsampled, distribution_full):
    question_types = ['conjunction', 'disjunction', 'existence', 'count']
    
    # Custom sort function so 'yes' and 'no' are at the beginning.
    # And count 11 does not occur before 2 (it was: 0, 1, 11, 2, 3).
    def custom_sort(value):
        try:
            return int(value)
        except ValueError:
            return float('-inf')  
    
    answers = sorted(distribution_downsampled.columns.tolist(), key=custom_sort) 

    # Create the figure.
    fig, ax = plt.subplots(figsize=(14, 8))
    bar_width = 0.35

    # Define the colors for the labels.
    named_colours = sns.color_palette("hsv", len(answers))
    answer_to_color = dict(zip(answers, named_colours))

    # Create bars.
    index = 0
    for question_type in question_types:
        if question_type in distribution_downsampled.index and question_type in distribution_full.index:

            # Get counts for the downsampled and normal questions.
            counts_downsampled = map(
                lambda x: x * 100,
                distribution_downsampled.loc[question_type, answers].tolist()
                )
            counts_full = map(
                lambda x: x * 100,
                distribution_full.loc[question_type, answers].tolist()
                )

            # Create bars for the downsampled part.
            bottom = 0
            for answer, count in zip(answers, counts_downsampled):
                ax.bar(
                    index - bar_width/2, count, bar_width, bottom=bottom,
                    label=answer if index == 0 else "",
                    color=answer_to_color[answer], alpha=0.5
                    )
                bottom += count 

            # Create bars for the not downsampled part.
            bottom = 0
            for answer, count in zip(answers, counts_full):
                ax.bar(
                    index + bar_width/2, count, bar_width, bottom=bottom,
                    label="" if index != 0 else "",
                    color=answer_to_color[answer]
                    )
                bottom += count

            index += 1

    # Set the labels, ticks and ranges for the axes.
    ax.set_xlabel("Question Types")
    ax.set_ylabel("Proportion of Answers (%)")
    # ax.set_title("Answer Distributions Before and After Downsampling")
    ax.set_xticks(range(len(question_types)))
    ax.set_xticklabels([f"{qt}" for qt in question_types])
    ax.set_ylim(0, 100)

    # Add legend.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) 
    ax.legend(
        by_label.values(), by_label.keys(), loc="upper left",
        bbox_to_anchor=(1.05, 1), ncol=2
        )

    # Show the figure.
    plt.tight_layout()
    plt.show()


def main():
    question_types = ["count", "conjunction", "disjunction", "existence"]

    # Load the questions (both downsampled and not).
    df_questions = pd.read_csv("../questions/intermediate_files/questions.csv")
    df_questions_downsampled = pd.read_csv("../questions/questions_downsampled.csv")

    # Extract the basic question types.
    df_questions["main_question_type"] = df_questions["question_type"].apply(lambda x: x.split('_')[0])
    df_questions_downsampled["main_question_type"] = df_questions_downsampled["question_type"].apply(lambda x: x.split('_')[0])

    # Filter relevant question types.
    df_filtered = df_questions[df_questions["main_question_type"].isin(question_types)]
    df_filtered_downsampled = df_questions_downsampled[df_questions_downsampled["main_question_type"].isin(question_types)]

    distribution = get_distribution(df_filtered)
    distribution_downsampled = get_distribution(df_filtered_downsampled)
    plot_combined_distribution(distribution, distribution_downsampled)

if __name__ == "__main__":
    main()
