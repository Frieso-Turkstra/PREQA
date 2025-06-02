import pandas as pd

df_questions = pd.read_csv("data/questions_downsampled.csv")
df_model_input = pd.DataFrame()

df_model_input["question"] = df_questions["question"]
df_model_input["question_id"] = df_questions["uid"]
df_model_input["image"] = df_questions["uid"].apply(lambda x: f"{x}_view.jpg")

df_model_input.to_csv("model_inputs/path_to_optimal_view.csv", index=False)
