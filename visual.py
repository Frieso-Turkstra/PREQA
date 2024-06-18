import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

df = pd.read_csv("data/questions.csv")

totals = Counter(df.question_type)
totals.pop("logical_room")
totals.pop("logical")

# Existence questions
existence_yes = len(df[(df.question_type == "existence") & (df.answer == "yes")])
existence_no = totals["existence"] - existence_yes

existence_room_yes = len(df[(df.question_type == "existence_room") & (df.answer == "yes")])
existence_room_no = totals["existence_room"] - existence_room_yes

# Count questions
count_df = df[df.question_type == "count"]
count_df["answer"] = count_df.answer.astype(int)
count_0 = len(count_df[count_df.answer == 0])
count_1 = len(count_df[count_df.answer == 1])
count_2 = len(count_df[count_df.answer > 1])

count_room_df = df[df.question_type == "count_room"]
count_room_df["answer"] = count_room_df.answer.astype(int)
count_room_0 = len(count_room_df[count_room_df.answer == 0])
count_room_1 = len(count_room_df[count_room_df.answer == 1])
count_room_2 = len(count_room_df[count_room_df.answer > 1])


# # Location questions
location_office = len(df[df.question_type == "location"].apply(lambda x: "office" in x.answer, axis=1))
location_hallway = len(df[df.question_type == "location"].apply(lambda x: "hallway" in x.answer, axis=1))
location_living_room = len(df[df.question_type == "location"].apply(lambda x: "living room" in x.answer, axis=1))

# Colour questions
colour_df = df[df.question_type == "colour"]
print(Counter(df.answer))

# Preposition questions

# Logical questions

# Plotting
fig, ax = plt.subplots()

# Bar widths
bar_width = 0.75  # Width of the bars

# Bars for Existence
bars_existence_yes = ax.bar(1, existence_yes, bar_width, label='Yes', color='blue')
bars_existence_no = ax.bar(1, existence_no, bar_width, label='No', bottom=existence_yes, color='orange')

bars_existence_room_yes = ax.bar(2, existence_room_yes, bar_width, color='blue')
bars_existence_room_no = ax.bar(2, existence_room_no, bar_width, bottom=existence_room_yes, color='orange')

bars = []
# Bars for Location
bottom = 0
for category, count in zip(["office", "hallway", "living room"], [location_office, location_hallway, location_living_room]):
    bar = ax.bar(3, count, bar_width, bottom=bottom, label=category)
    bars.append(bar)
    bottom += count  # Update bottom position for the next stacked bar

# Bars for Count
bottom = 0
for category, count in zip(["0", "1", "2+"], [count_0, count_1, count_2]):
    bar = ax.bar(4, count, bar_width, bottom=bottom, label=category)
    bars.append(bar)
    bottom += count  # Update bottom position for the next stacked bar

# Bars for Room Count
bottom = 0
for count in [count_room_0, count_room_1, count_room_2]:
    bar = ax.bar(5, count, bar_width, bottom=bottom)
    bars.append(bar)
    bottom += count  # Update bottom position for the next stacked bar

# X-axis labels
ax.set_xlabel('Question Types')
ax.set_ylabel('Number of Questions')
ax.set_title('Number of Questions by Type')

# X-axis ticks and labels
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(['Existence', 'Existence_room', 'Location', 'Count', 'Count_room'])

# Y-axis range
ax.set_ylim(0, max(totals.values()) + 10)

# Add counts above the bars
def add_counts(ax, bars):
    for bar in bars:
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_y() + bar.get_height() / 2
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(x, y),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_counts(ax, bars_existence_yes)
add_counts(ax, bars_existence_no)
add_counts(ax, bars_existence_room_yes)
add_counts(ax, bars_existence_room_no)
for bar in bars:
    add_counts(ax, bar)

# Add legend
ax.legend()

plt.tight_layout()
plt.show()
