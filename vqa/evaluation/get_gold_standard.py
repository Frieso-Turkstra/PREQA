import pandas as pd
import inflect

# Used to convert numbers to words.
p = inflect.engine()

# Read in the questions and annotation files.
df = pd.read_csv("../../questions/questions_downsampled.csv")
objects_df = pd.read_csv("../../annotations/annotations.csv")

# Extract the answers.
answers = []
for _, row in df.iterrows():
    question_id = row["uid"]
    question_type = row["question_type"]

    if question_type.split("_")[0] in ("existence", "conjunction", "disjunction"):
        # Answer is yes or no and can directly be taken from the file.
        answer = row["answer"]

    if question_type.startswith("count"):
        # Save count answers as their string counterpart (1 -> "one").
        answer = p.number_to_words(row["answer"])

    if question_type.startswith("location"):
        # Get all unique locations.
        answer_dictionary = eval(row["answer"])
        answer = list(set(answer_dictionary.values()))

    if question_type.startswith("colour"):
        # Get all unique colours.
        answer_dictionary = eval(row["answer"])
        answer = list({colour for colours in answer_dictionary.values() for colour in colours})
    
    if question_type.startswith("spatial"):
        # Get the labels for the object ids.
        answer_dictionary = eval(row["answer"])
        object_ids = [object_id for object_ids in answer_dictionary.values() for object_id in object_ids]
        labels = {objects_df.loc[objects_df.object_id == object_id, "label"].tolist()[0] for object_id in object_ids}
        answer = list(labels)

    answers.append(answer)

# Save results.
gold_df = pd.DataFrame()
gold_df["question_id"] = df["uid"]
gold_df["answer"] = answers
gold_df.to_csv("gold_standard.csv", index=False)
