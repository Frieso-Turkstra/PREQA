import pandas as pd
import os

# Read in the file of which a sample will be manually correct.
eval_file = f"intermediate_files/eval_phi3_text_only.csv"
df = pd.read_csv(eval_file)

# Read in the file with all the questions.
questions_file = "../../questions/questions_downsampled.csv"
questions_df = pd.read_csv(questions_file)

# Get a subsample where each question type occurs equally frequent.
sample_size = 100
question_types = df["question_type"].unique()
samples_per_type = sample_size // len(question_types)

sampled_df = pd.concat([
    df[df.question_type == q_type].sample(samples_per_type, replace=True)
    for q_type in question_types
])

# Randomly sample to fill up to sample size.
if len(sampled_df) < sample_size:
    additional_samples = df[~df.index.isin(sampled_df.index)].sample(sample_size - len(sampled_df))
    sampled_df = pd.concat([sampled_df, additional_samples])
    
# Manually correct every prediction.
manual_scores = []
count = 0
total = len(sampled_df)
for _, row in sampled_df.iterrows():
    question_id = row["question_id"]
    prediction = row["prediction"]
    answer = row["answer"]
    question = questions_df.loc[questions_df.uid == question_id, "question"].squeeze()

    os.system("cls")
    print(f"{count+1}/{total}")
    print("Question:", question)
    print("Prediction:", prediction)
    print("Answer:", answer)
    score = input("Does the prediction contain the right answer? (y/n) ")

    manual_scores.append(score.startswith("y"))
    count += 1

# Save the manual corrections.
sampled_df["manual_score"] = manual_scores
sampled_df.to_csv(f"intermediate_files/eval_phi3_text_only_sampled_2.csv", index=False)
