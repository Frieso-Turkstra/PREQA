from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
import argparse
import torch
import csv

torch.random.manual_seed(0)

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--questions_file",
                        help="File with all the questions",
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
    model_id = "microsoft/Phi-3-mini-128k-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Write each row continuously to a csv file.
    with open(args.output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["question_id", "prediction"])
        writer.writeheader()
        
        # For each question.
        for index, row in data.iterrows():
        
            question_id = row["question_id"]
            question = row["question"]

            # Create the prompt.
            messages = [{"role": "user", "content": question}]
            
            # Prompt the model.
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            
            generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.7,
                "do_sample": True,
            }
            
            output = pipe(messages, **generation_args)
            response = output[0]['generated_text']
            
            # Save result to csv file.
            result = {"question_id": question_id, "prediction": response}
            writer.writerow(result)


if __name__ == "__main__":
    main()
