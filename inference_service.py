# inference_service.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
from spam_detection import load_model_from_hub, classify_comment

app = FastAPI()

# Load the model and tokenizer only once when the service starts
hf_token = os.getenv("HF_TOKEN")
repo_name = "bluepika2/youtube-spam-detection"  # Replace with your actual repository name
model, tokenizer = load_model_from_hub(repo_name, use_auth_token=hf_token)

class InferenceRequest(BaseModel):
    comment_text: str

@app.post("/classify")
def classify_comment_endpoint(request: InferenceRequest):
    prediction = classify_comment(request.comment_text, model, tokenizer)
    return {"prediction": prediction}
