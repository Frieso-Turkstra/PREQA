from Model import Model
from typing import Optional, Union
import pandas as pd

class MetaModel(Model, model_key="meta"):
    def request_body(self, prompt: str, temperature: float=0.5, top_p: float=0.9, max_gen_len: int=512) -> dict:
        return {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_gen_len": max_gen_len
        }
    
    @classmethod
    def extract_output(cls, model_outputs_df: pd.DataFrame, size: str) -> pd.Series:
        return model_outputs_df['modelOutput'].apply(lambda x: x['generation'].strip()[0])

    def format_prompt(
        self,
        message: Union[str, list[dict]],
        system_prompt: Optional[str] = None,
    ) -> str:
        prompt = "<|begin_of_text|>"

        # Add system instructions
        if system_prompt is not None:
            prompt += f"<|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|>"

        # Add in-context examples
        if isinstance(message, list):
            for msg in message:
                text = msg["content"][0]["text"]
                prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>{text}<|eot_id|>"
        else:
            prompt += f"<|start_header_id|>user<|end_header_id|>{message}<|eot_id|>"

        # Add assistant token to start generation
        prompt += "<|start_header_id|>assistant<|end_header_id|>"

        return prompt
