import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Load your data
eval_file = "eval_files/eval_phi3_text_only_sampled.csv"
df = pd.read_csv(eval_file)

# Generate a range of possible thresholds
thresholds = np.linspace(0, 1, 100)

# Store results
results = []

# Evaluate each threshold
for threshold in thresholds:
    # predicted_bool = df['word_similarity'] >= threshold
    predicted_bool = df["exact_match"]
    accuracy = accuracy_score(df['manual_score'], predicted_bool)
    results.append((threshold, accuracy))

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(results, columns=['Threshold', 'Accuracy'])

# Find the best threshold
best_threshold = results_df.loc[results_df['Accuracy'].idxmax()]

print(f"Best Threshold: {best_threshold['Threshold']}")
print(f"Accuracy at Best Threshold: {best_threshold['Accuracy']}")

# Optionally, plot the results
import matplotlib.pyplot as plt

plt.plot(results_df['Threshold'], results_df['Accuracy'])
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.title('Threshold vs. Accuracy')
plt.grid(True)
plt.show()
