import boto3
import json


prompt_data="financial statemnt analysis"
model_id="anthropic.claude-3-sonnet-20240229-v1:0"

bedrock=boto3.client(service_name="bedrock-runtime",region_name="us-east-1")

payload={
 "max_tokens": 256,
  "messages": [{"role": "user", "content": "What is the capital of India"}],
  "anthropic_version": "bedrock-2023-05-31"
}

body=json.dumps(payload)



# bedrock = boto3.client(service_name="bedrock-runtime")
# body = json.dumps({
#   "max_tokens": 256,
#   "messages": [{"role": "user", "content": "Hello, world"}],
#   "anthropic_version": "bedrock-2023-05-31"
# })

response = bedrock.invoke_model(body=body, modelId=model_id)

response_body = json.loads(response.get("body").read())
print(response_body.get("content"))
