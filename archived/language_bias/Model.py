import pandas as pd
import json
from tqdm import tqdm
from datetime import datetime
import time
from botocore.exceptions import ClientError
tqdm.pandas()

class Model:
    _registry = {}

    def __new__(cls, s3_client, bedrock_client, bedrock_runtime_client, sagemaker_runtime_client, id, *args, **kwargs):
        if "deepseek" in id:
            model_key = "deepseek"
        else:
            model_key = id.split(".")[0]
        if cls is Model:
            if model_key not in cls._registry:
                raise ValueError(f"Unknown model_key: {model_key}")
            subclass = cls._registry[model_key]
            return super(Model, subclass).__new__(subclass)
        return super(Model, cls).__new__(cls)

    def __init_subclass__(cls, model_key=None):
        super().__init_subclass__()
        if model_key is not None:
            Model._registry[model_key] = cls
        
    def __init__(self, s3_client, bedrock_client, bedrock_runtime_client, sagemaker_runtime_client, id):
        self.s3_client = s3_client
        self.bedrock_client = bedrock_client
        self.bedrock_runtime_client = bedrock_runtime_client
        self.sagemaker_runtime_client = sagemaker_runtime_client
        self.id = id

        self.s3_input_bucket = "batch.input.bucket"
        self.s3_output_bucket = "batch.output.bucket"
        self.s3_input_bucket_uri = "s3://batch.input.bucket/fallacy_detection/model_inputs.jsonl"
        self.s3_output_bucket_uri = "s3://batch.output.bucket/fallacy_detection/"
        self.service_role_arn = "arn:aws:iam::180294204696:role/service-role/ServiceRoleForBatchInference"

    def __str__(self):
        return f"Model(id={self.id})"

    @classmethod
    def extract_output(cls, model_outputs_df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def request_body(self) -> dict:
        raise NotImplementedError
    
    def run_inference(self, path, input_data: pd.DataFrame, batch_mode: bool) -> pd.DataFrame:
        output_file = path / "model_outputs.jsonl"
        if batch_mode:
            self.upload_to_s3(path, input_data)
            experiment_name = path.resolve().name
            job_arn = self.submit_batch_job(experiment_name)
            self.wait_for_batch_job(job_arn)
            job_id = job_arn.split('/')[-1]
            self.download_results(job_id, output_file)
            output_df = pd.read_json(output_file, lines=True)
            deduped_df = output_df.drop_duplicates(subset="recordId", keep="first")
            deduped_df.to_json(output_file, orient="records", lines=True)
        else:
            # with open(output_file, 'a') as f:
            #     for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
            #         result = self.invoke(row["modelInput"])
            #         row["modelOutput"] = result
            #         f.write(json.dumps(row.to_dict()) + "\n")

            input_data["modelOutput"] = input_data["modelInput"].progress_apply(self.invoke)
            input_data.to_json(output_file, orient="records", lines=True)

    def format_prompt(self, prompt: str, system_prompt: str) -> str:
        return prompt

    def invoke(self, model_input: dict) -> str:  
        try:
            response = self.bedrock_runtime_client.invoke_model(
                modelId=self.id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(model_input)
            )
            return json.loads(response["body"].read())
        except Exception as e:
            print(e)
            # response = self.sagemaker_runtime_client.invoke_endpoint(
            #     EndpointName="endpoint-deepseek-qwen-7b",
            #     ContentType='application/json',
            #     Accept='application/json',
            #     Body=json.dumps(model_input),
            # )
    
    # def converse(self, model_input: dict) -> str:
    #     response = self.bedrock_runtime_client.converse(
    #         modelId=self.id,
    #         messages=messages,
    #         system=system_prompts,
    #         inferenceConfig=inference_config,
    #         additionalModelRequestFields=additional_model_fields
    #     )
    #     return response['output']['message']
    
    def upload_to_s3(self, path, input_data: pd.DataFrame) -> None:
        input_data_file = path / "model_inputs.jsonl"
        input_data.to_json(input_data_file, orient="records", lines=True)
        s3_key = "fallacy_detection/model_inputs.jsonl"
        self.s3_client.upload_file(input_data_file, self.s3_input_bucket, s3_key)

    def submit_batch_job(self, experiment_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"{experiment_name}_{timestamp}"
        job_name = job_name.replace('_', '-')

        response = self.bedrock_client.create_model_invocation_job(
            jobName=job_name,
            roleArn=self.service_role_arn,
            modelId=self.id,
            inputDataConfig={
                's3InputDataConfig': {
                    's3Uri': self.s3_input_bucket_uri,
                    's3InputFormat': 'JSONL'
                }
            },
            outputDataConfig={
                's3OutputDataConfig': {
                    's3Uri': self.s3_output_bucket_uri
                }
            },
            timeoutDurationInHours=24
        )
        return response['jobArn']
    
    def wait_for_batch_job(self, job_arn):
        wait_time = 5
        while True:
            job_status = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
            status = job_status['status']
            print(f"Job status: {status}")
            if status in ['Completed', 'Failed']:
                break
            time.sleep(wait_time)
            wait_time = min(wait_time * 2, 60)  # Exponential backoff, max wait time of 2 minute

    def download_results(self, job_id, output_file):
        self.s3_client.download_file(
            "batch.output.bucket",
            f"fallacy_detection/{job_id}/model_inputs.jsonl.out",
            output_file
        )

    

    
