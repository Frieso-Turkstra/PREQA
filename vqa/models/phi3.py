from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor 
from PIL import Image 
import pandas as pd
import argparse
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
                        required=False,
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
    model_id = "microsoft/Phi-3-vision-128k-instruct" 
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cuda", trust_remote_code=True,
        torch_dtype="auto", _attn_implementation='eager'
        )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

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
            image = Image.open(image_file_path)
            
            # Create the prompt.
            messages = [{"role": "user", "content": f"<|image_1|>\n{question}"}] 
            
            prompt = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
                )
            
            # Prompt the model.
            inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
            generation_args = { 
                "max_new_tokens": 500, 
                "temperature": 0.7, 
                "do_sample": True, 
            } 
            generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
            
            # Remove the input tokens.
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
            
            # Save result to csv file.
            result = {"question_id": question_id, "prediction": response}
            writer.writerow(result)


if __name__ == "__main__":
    main()
