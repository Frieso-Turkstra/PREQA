from transformers import AutoModel, AutoTokenizer
from PIL import Image
import pandas as pd
import argparse
import torch
import csv
import os

torch.random.manual_seed(0)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--questions_file",
                        help="File with all the questions",
                        required=True,
                        type=str)
    parser.add_argument("-i", "--image_directory",
                        help="Directory with the stitched path images.",
                        required=True,
                        type=str)
    parser.add_argument("-o", "--output_file",
                        help="Path to the output file.",
                        required=True,
                        type=str)
    args = parser.parse_args()
    return args
    

def main():
    # Read in the command-line arguments and questions file.
    args = create_arg_parser()
    data = pd.read_csv(args.questions_file)

    # Initialize the model.
    model_id = "openbmb/MiniCPM-Llama3-V-2_5"
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
        )
    model = model.to(device='cuda')

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()

    # Write each row continuously to a csv file.
    with open(args.output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["question_id", "prediction"])
        writer.writeheader()

        # For each question.
        for index, row in data.iterrows():
            
            question_id = row["question_id"]
            question = row["question"]
            image_file = row["image"]
            
            image_file_path = os.path.join(args.image_directory, image_file)
            image = Image.open(image_file_path).convert('RGB')
            
            # Create the prompt.
            msgs = [{'role': 'user', 'content': question}]
            
            # Prompt the model.
            response = model.chat(
                image=image,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7,
            )
            
            # Save result to csv file.
            result = {"question_id": question_id, "prediction": response}
            writer.writerow(result)
        

if __name__ == "__main__":
    main()
