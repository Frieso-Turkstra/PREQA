import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import pandas as pd
import ast

# Graph settings.
fig, ax = plt.subplots()
bar_width = 0.75  
named_colours = list(mcolors.CSS4_COLORS)

# Read in the data.
df = pd.read_csv("data/questions.csv")

# Merge type_room and type_object
basic_question_types = ["existence", "count", "location", "colour", "spatial", "conjunction", "disjunction"]
for question_type in basic_question_types:
    df["question_type"] = df["question_type"].apply(
        lambda x: question_type if x in (f"{question_type}_room", f"{question_type}_object") else x
        )

totals = Counter(df.question_type)


# Set question_type (existence and count and conjunction and disjunction)
question_type = "location"
question_types = [question_type, f"{question_type}_location", f"{question_type}_colour", f"{question_type}_colour_location"]
answers = sorted([df[df.question_type == question_type].answer.unique().tolist() for question_type in question_types], reverse=True)

question_types = ["location_object", "location_colour"]

bars = []
for i, (question_type, labels) in enumerate(zip(question_types, answers)):

    # Get counts.
    counts = []
    for label in labels:
        counts.append(len(df[(df.question_type == question_type) & (df.answer == label)]) / totals[question_type] * 100)

    # Create bars.
    bottom = 0
    start = 10
    step = int((len(named_colours) - start) / len(labels))
    colours = {labels[i]: named_colours[start+i*step] for i in range(len(labels))}
    for label, count in zip(labels, counts):
        if i == 0:
            bars.append(ax.bar(i, count, bar_width, bottom=bottom, label=label, color=colours[label]))
        else:
            bars.append(ax.bar(i, count, bar_width, bottom=bottom, color=colours[label]))
        bottom += count  

# Set x-axis labels.
ax.set_xlabel("Question Types")
ax.set_ylabel("Number of Questions")
ax.set_title("Number of Questions by Type")

# Set X-axis ticks and labels.
ax.set_xticks(range(len(question_types)))
ax.set_xticklabels(question_types)

# Set Y-axis range.
ax.set_ylim(0, 100)

# Add legend
ax.legend()

plt.tight_layout()
plt.show()
