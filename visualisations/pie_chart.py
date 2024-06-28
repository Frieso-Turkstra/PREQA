from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file",
                        help="File with the questions.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def merge_prefixes(counter, delimiter="_"):
    # Combine all the counts for each question type variant.
    merged_counter = defaultdict(int)
    for key, count in counter.items():
        prefix = key.split(delimiter)[0]
        merged_counter[prefix] += count
    return Counter(merged_counter)

def autopct_func(pct):
    return f"{pct:.1f}%" if pct > 0.5 else ""


def main():
    # Read in the command-line arguments and questions file.
    args = create_arg_parser()
    df = pd.read_csv(args.input_file)

    # Count how often each question type occurs.abs
    data = merge_prefixes(Counter(df.question_type))

    # Get the labels and sizes.
    labels = list(data.keys())
    sizes = list(data.values())

    # Create the pie chart.
    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes, autopct=autopct_func, startangle=90,
        pctdistance=0.85, textprops=dict(color="w")
        )

    # Adjust text labels to prevent overlap.
    for text in texts:
        text.set_fontsize(10)
        text.set_color("black")

    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color("white")

    # Draw circle at center to make it look like a donut.
    centre_circle = plt.Circle((0, 0), 0.70, fc="white")
    fig.gca().add_artist(centre_circle)

    # Set aspect ratio so pie is drawn as a circle.
    ax.axis("equal")  

    # Add legend and title.
    plt.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.title("Question Type Proportions Before and After Downsampling")

    # Show the image.
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
