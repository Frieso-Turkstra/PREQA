import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def evaluate_file(file_path, thresholds):
    df = pd.read_csv(file_path)
    results = []
    for threshold in thresholds:
        predicted_bool = df["word_similarity"] >= threshold
        accuracy = accuracy_score(df["manual_score"], predicted_bool)
        results.append((threshold, accuracy))
    return pd.DataFrame(results, columns=["Threshold", "Agreement"])


# Check 100 possible thresholds.
thresholds = np.linspace(0, 1, 100)

# Evaluate the minicpm file.
eval_file_1 = "intermediate_files/eval_minicpm_sampled.csv"
results_df1 = evaluate_file(eval_file_1, thresholds)

# Evaluate the phi3 file.
eval_file_2 = "intermediate_files/eval_phi3_sampled.csv"
results_df2 = evaluate_file(eval_file_2, thresholds)

# Plot the results for both files.
plt.plot(results_df1["Threshold"], results_df1["Agreement"] * 100, label="MiniCPM-Llama3-V 2.5")
plt.plot(results_df2["Threshold"], results_df2["Agreement"] * 100, label="Phi-3-vision-128k-instruct")

# Add a vertical line at the optimal threshold.
optimal_threshold = 0.626
plt.axvline(x=optimal_threshold, color="r", linestyle="--", label="Optimal Threshold")

# Show plot.
plt.xlabel("Cosine Similarity Threshold")
plt.ylabel("Agreement (%)")
plt.title("Threshold vs. Agreement")
plt.grid(True)
plt.legend()
plt.show()
