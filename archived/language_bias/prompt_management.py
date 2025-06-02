from botocore.exceptions import ClientError
import logging
import boto3
from typing import Optional
import pprint
from functools import cached_property
from jinja2 import Template


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Prompt:
    def __init__(
            self,
            client: boto3.Session.client,
            name: Optional[str] = None,
            id: Optional[str] = None,
            version: str="latest",
            variant_name: Optional[str] = None
        ):
        if name is None and id is None:
            raise ValueError("Either 'name' or 'id' must be provided to fetch the prompt.")
        
        self.client = client
        self.variant_name = variant_name
        self.prompt = self.fetch_prompt(name, id, version)

    def __str__(self):
        return f"Prompt(name={self.name}, id={self.id}, version={self.version})"
    
    def __repr__(self):
        return pprint.pformat(self.prompt, indent=2)
        
    @property
    def name(self) -> str:
        return self.prompt.get("name")
    
    @property
    def id(self) -> str:
        return self.prompt.get("id")
    
    @property
    def version(self) -> str:
        return self.prompt.get("version")
    
    @cached_property
    def variant(self) -> dict:
        if self.variant_name is None:
            self.variant_name = self.prompt.get("defaultVariant")

        for variant in self.prompt.get("variants", []):
            if variant.get("name") == self.variant_name:
                return variant

        raise ValueError(f"Variant '{self.variant_name}' not found in prompt '{self.name}'.")
    
    @property
    def model_id(self) -> str:
        return self.variant.get("modelId")
    
    @property
    def template_type(self) -> str:
        return self.variant.get("templateType")
    
    @property
    def template_configuration(self) -> dict:
        return self.variant.get("templateConfiguration").get(self.template_type.lower())
    
    @property
    def input_variables(self) -> list:
        return [var['name'] for var in self.template_configuration.get("inputVariables")]

    @property
    def template(self) -> str:
        key = "messages" if self.template_type == "CHAT" else "text"
        return self.template_configuration.get(key)
    
    @property
    def inference_configuration(self) -> dict:
        return self.variant.get("inferenceConfiguration").get("text")
    
    @property
    def system(self) -> Optional[str]:
        system_prompt = self.template_configuration.get("system", {})
        if system_prompt:
            return system_prompt[0].get("text")


    def fetch_prompt(self, name: Optional[str], id: Optional[str], version: str) -> dict:
        """
        Resolve prompt ID and version, then retrieve the prompt object.
        """

        if not id:
            logger.debug(f"No prompt ID provided. Fetching prompt ID by name: {name}")
            id = self.find_prompt_id_by_name(name)
        
        resolved_version = self.prepare_version(version, id)

        logger.info("Fetching prompt with ID=%s and version=%s", id, resolved_version)
        return self.client.get_prompt(promptIdentifier=id, promptVersion=resolved_version)


    def find_prompt_id_by_name(self, name: str) -> str:
        response = self.client.list_prompts()
        prompts = response.get("promptSummaries", []) 

        for prompt in prompts:
            if prompt.get("name") == name:
                return prompt.get("id")
        
        raise ValueError(f"Prompt with name '{name}' not found.")
    

    def prepare_version(self, version: str, id: str) -> str:
        # Calling list_prompts with a specified id returns all versions of that prompt 
        response = self.client.list_prompts(promptIdentifier=id)
        prompt_summaries = response.get("promptSummaries", [])
        versions = [prompt["version"] for prompt in prompt_summaries]

        if version == "latest":
            numeric_versions = [int(v) for v in versions if v.isdigit()]
            if not numeric_versions:
                raise ValueError("No numeric versions found for prompt ID '%s'.", id)
            version = str(max(numeric_versions))

        elif version == "new":
            version = self.create_version(id)

        elif not version in versions:
            raise ValueError(f"Prompt version '{version}' not found for prompt ID '{id}'.")
        
        return version
        

    def create_version(self, id: str, description: Optional[str] = None) -> str:
        """
        Creates a version of an Amazon Bedrock managed prompt.

        Args:
        client: Amazon Bedrock Agent boto3 client.
        prompt_id (str): The identifier of the prompt to create a version for.
        description (str, optional): A description for the version.

        Returns:
            str: The created version number.
        """
        try:
            logger.info("Creating version for prompt ID: %s.", id)
            
            create_params = {
                "promptIdentifier": id
            }
            
            if description:
                create_params["description"] = description
                
            response = self.client.create_prompt_version(**create_params)

            logger.info("Successfully created prompt version: %s", response["version"])
            logger.info("Prompt version ARN: %s", response["arn"])

            return response["version"]


        except ClientError as e:
            logger.exception("Client error creating prompt version: %s", str(e))
            raise

        except Exception as e:
            logger.exception("Unexpected error creating prompt version: %s", str(e))
            raise

    
    def render(self, input_data: list[str]) -> str:
        if len(input_data) != len(self.input_variables):
            raise ValueError("Input data does not match required input variables.")
        
        context = dict(zip(self.input_variables, input_data))
        
        if self.template_type == "CHAT":

            rendered_messages = []
            for message in self.template:
                content = [
                        {"text": Template(part["text"]).render(context)}
                        for part in message.get("content", [])
                    ]
                if self.model_id == "mistral.mistral-large-2407-v1:0":
                    part = message.get("content")[0]
                    content = Template(part["text"]).render(context)

                rendered_message = {
                    "role": message["role"],
                    "content": content
                }
                rendered_messages.append(rendered_message)
            return rendered_messages
        
        return Template(self.template).render(context)
