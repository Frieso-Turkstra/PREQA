import pandas as pd
import boto3
from pathlib import Path
from prompt_management import Prompt
from Model import Model
from MetaModel import MetaModel


batch_mode = True
path = Path("language_bias/results")

data = pd.read_csv("PREQA/questions_downsampled.csv")

session = boto3.Session(region_name="us-west-2")
bedrock_client = session.client("bedrock")
bedrock_agent_client = session.client("bedrock-agent")
bedrock_runtime_client = session.client("bedrock-runtime")
s3_client = session.client("s3")
print("Initialized AWS clients.")

prompt = Prompt(bedrock_agent_client, id=None, name="EQA_language_bias", version="latest")
print(f"Loaded prompt: {prompt}")

model = Model(s3_client, bedrock_client, bedrock_runtime_client, None, prompt.model_id)
print(f"Loaded model: {model}")

# Get model inputs
model_inputs = []
inputs = data["question"]

for entry in inputs:
    rendered = prompt.render([entry])
    formatted = model.format_prompt(rendered, prompt.system)
    body = model.request_body(formatted)

    model_inputs.append(body)
    
model_inputs_df = pd.DataFrame({"modelInput": model_inputs, "recordId": data["uid"]})
model_inputs_df.to_json(path / 'model_inputs.jsonl', orient="records", index=False, lines=True)
print(f"Successfully prepared {len(model_inputs_df)} model_inputs.")

confirmation = input("is it okay?")
if confirmation.lower() != 'yes':
    print("Exiting without running inference.")
    exit()

# Save model outputs
model.run_inference(path, model_inputs_df, batch_mode)
print(f"Inference complete. Results are saved to {path / 'model_outputs.jsonl'}")
