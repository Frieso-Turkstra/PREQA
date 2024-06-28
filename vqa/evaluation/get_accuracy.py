import pandas as pd

# Determine which file to calculate the accuracy scores for.
model = "minicpm"
variant = "forward_only"
df = pd.read_csv(f"eval_files/eval_{model}_{variant}.csv")

# If the similarity is above these thresholds, the answer is considered correct.
WORD_THRESHOLD = 0.626
SENTENCE_THRESHOLD = 0.15

# Calculate accuracy scores.
accuracy_results = []
exact_match_accuracy = len(df[df["exact_match"]]) / len(df)
word_similarity_accuracy = len(df[df["word_similarity"] > WORD_THRESHOLD]) / len(df)
sentence_similarity_accuracy = len(df[df["sentence_similarity"] > SENTENCE_THRESHOLD]) / len(df)

accuracy_results.append({
    "question_type": "overall",
    "exact_match_accuracy": exact_match_accuracy,
    "word_similarity_accuracy": word_similarity_accuracy,
    "sentence_similarity_accuracy": sentence_similarity_accuracy
})

# Also calculate accuracy per question type.
for question_type, group in df.groupby("question_type"):
    exact_match_accuracy = len(group[group["exact_match"]]) / len(group)
    word_similarity_accuracy = len(group[group["word_similarity"] > WORD_THRESHOLD]) / len(group)
    sentence_similarity_accuracy = len(group[group["sentence_similarity"] > SENTENCE_THRESHOLD]) / len(group)
    
    accuracy_results.append({
        "question_type": question_type,
        "exact_match_accuracy": exact_match_accuracy,
        "word_similarity_accuracy": word_similarity_accuracy,
        "sentence_similarity_accuracy": sentence_similarity_accuracy
    })

# Save results.
accuracy_df = pd.DataFrame(accuracy_results)
output_file = f"accuracy_scores/scores_{model}_{variant}.csv"
accuracy_df.to_csv(output_file, index=False)
