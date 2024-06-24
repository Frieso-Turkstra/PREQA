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
    # Get sentence embedding.
    return model.encode(sentence, convert_to_tensor=True)

def is_similar_sentence(prediction: str, answer: str, question_type: str) -> float:
    prediction_embedding = get_sentence_embedding(prediction)

    if question_type.startswith(("location", "colour", "spatial")):
        answers = eval(answer)
        similarities = [util.pytorch_cos_sim(prediction_embedding, get_sentence_embedding(answer)).item() for answer in answers]
        similarity = max(similarities)
    else:
        answer_embedding = get_sentence_embedding(answer)
        similarity = util.pytorch_cos_sim(prediction_embedding, answer_embedding).item()
    return similarity

def is_similar_word(prediction: str, answer: str, question_type: str) -> float:
    prediction = prediction.split()
    if question_type.startswith(("location", "colour", "spatial")):
        answers = eval(answer)
        similarities = [util.pytorch_cos_sim(get_sentence_embedding(word), get_sentence_embedding(answer)).item() for word, answer in itertools.product(prediction, answers)]
    else:
        answer_embedding = get_sentence_embedding(answer)
        similarities = [util.pytorch_cos_sim(get_sentence_embedding(word), answer_embedding).item() for word in prediction]
    return max(similarities)

def is_exact_match(prediction: str, answer: str, question_type: str) -> bool:
    if question_type.startswith(("location", "colour", "spatial")):
        answers = eval(answer)
        return any([answer in prediction for answer in answers])
    return answer in prediction


# predictions file will have columns "question_id" and "prediction".
# gold standard file has columns "question_id" and "answer".
args = create_arg_parser()
pred_df = pd.read_csv(args.predictions_file)
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

# Calculate overall accuracy
# THRESHOLD = 0.7
exact_match_accuracy = len(df[df["exact_match"]]) / len(df)
print("Exact match accuracy: ", exact_match_accuracy)

for i in range(1, 10):
    sentence_similarity_accuracy = len(df[df["sentence_similarity"] > (i / 10)]) / len(df)
    word_similarity_accuracy = len(df[df["word_similarity"] > (i / 10)]) / len(df)
    print("Threshold set to:", i / 10)
    print("Sentence similarity accuracy:", sentence_similarity_accuracy)
    print("Word similarity accuracy:", word_similarity_accuracy)


df.to_csv(args.output_file, index=False)

# TODO evaluate on a general and class-level (i.e., per question type VARIANT)
