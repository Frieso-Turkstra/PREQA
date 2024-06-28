import pandas as pd
import os

# dfs = []
# for variant in ("random", "forward_only", "object", "optimal_view"):
eval_file = f"eval_files/eval_phi3_text_only.csv"
questions_file = "../data/questions_downsampled.csv"
df = pd.read_csv(eval_file)
questions_df = pd.read_csv(questions_file)
seen_df = pd.read_csv("eval_files/eval_phi3_text_only_sampled.csv")

sample_size = 50
question_types = df["question_type"].unique()
samples_per_type = sample_size // len(question_types)

df = df[~df.isin(seen_df).all(axis=1)]
        
sampled_df = pd.concat([
    df[df.question_type == q_type].sample(samples_per_type, replace=True)
    for q_type in question_types
])

if len(sampled_df) < sample_size:
    additional_samples = df[~df.index.isin(sampled_df.index)].sample(sample_size - len(sampled_df))
    sampled_df = pd.concat([sampled_df, additional_samples])
    
# dfs.append(sampled_df)

# sampled_df = pd.concat(dfs)
# print("should be 100:", len(sampled_df))


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

sampled_df["manual_score"] = manual_scores
sampled_df.to_csv(f"eval_files/eval_phi3_text_only_sampled_2.csv", index=False)
