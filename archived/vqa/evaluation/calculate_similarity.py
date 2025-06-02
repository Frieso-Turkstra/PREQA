"""
This script calculates the cosine similarity scores between the prediction 
and the correct answer. Both on the word and entire prediction level.
Additionally, the prediction is checked for an exact match with the answer.
"""

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import itertools
import argparse
import string


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions_file",
                        help="File with all the predictions.",
                        required=True,
                        type=str)
    parser.add_argument("-g", "--gold_standard_file",
                        help="File with the ground truth answers.",
                        required=True,
                        type=str)
    parser.add_argument("-q", "--questions_file",
                        help="File with the questions.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_file",
                        help="Path to the output file.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args

def preprocess(text: str) -> str:
    # Make lowercase and remove punctuation.
    return text.lower().translate(str.maketrans('', '', string.punctuation))

def get_sentence_embedding(sentence: str):
    return model.encode(sentence, convert_to_tensor=True)

def is_similar_sentence(prediction: str, answer: str, question_type: str) -> float:
    # Checks the cosine similarity between the answer and the entire prediction.
    prediction_embedding = get_sentence_embedding(prediction)

    if question_type.startswith(("location", "colour", "spatial")):
        # Check for each of the possible answers.
        answers = eval(answer)
        similarities = [util.pytorch_cos_sim(prediction_embedding, get_sentence_embedding(answer)).item() for answer in answers]
        similarity = max(similarities)
    else:
        answer_embedding = get_sentence_embedding(answer)
        similarity = util.pytorch_cos_sim(prediction_embedding, answer_embedding).item()
    return similarity

def is_similar_word(prediction: str, answer: str, question_type: str) -> float:
    # Checks the cosine similarity between the answer and each word in the prediction.
    prediction = prediction.split()
    if question_type.startswith(("location", "colour", "spatial")):
        # Check for each of the possible answers.
        answers = eval(answer)
        similarities = [util.pytorch_cos_sim(get_sentence_embedding(word), get_sentence_embedding(answer)).item() for word, answer in itertools.product(prediction, answers)]
    else:
        answer_embedding = get_sentence_embedding(answer)
        similarities = [util.pytorch_cos_sim(get_sentence_embedding(word), answer_embedding).item() for word in prediction]
    return max(similarities, default=0.0)

def is_exact_match(prediction: str, answer: str, question_type: str) -> bool:
    # Checks if the answer is letter for letter present in the prediction.
    if question_type.startswith(("location", "colour", "spatial")):
        answers = eval(answer)
        return any([answer in prediction for answer in answers])
    return answer in prediction


# Predictions file will have columns "question_id" and "prediction".
# Gold standard file has columns "question_id" and "answer".
args = create_arg_parser()

# pred_df = pd.read_csv(args.predictions_file)
output_df = pd.read_json(args.predictions_file, lines=True)
predictions = output_df['modelOutput'].apply(lambda x: x['generation'].strip())
record_ids = output_df['recordId']
pred_df = pd.DataFrame({
    'question_id': record_ids,
    'prediction': predictions
})

gold_df = pd.read_csv(args.gold_standard_file)
questions_df = pd.read_csv(args.questions_file)
df = pd.merge(pred_df, gold_df, on="question_id")

df["prediction"] = df["prediction"].fillna('').apply(preprocess)
df["question_type"] = df["question_id"].apply(lambda x: questions_df.loc[questions_df.uid == x.split("_")[0], "question_type"].iloc[0])

# Load the pre-trained model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Get similarity and exact match scores.
df["sentence_similarity"] = df.apply(lambda x: is_similar_sentence(x.prediction, x.answer, x.question_type), axis=1)
df["word_similarity"] = df.apply(lambda x: is_similar_word(x.prediction, x.answer, x.question_type), axis=1)
df["exact_match"] = df.apply(lambda x: is_exact_match(x.prediction, x.answer, x.question_type), axis=1)

df.to_csv(args.output_file, index=False)
