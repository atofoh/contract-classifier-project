
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, re

app = FastAPI(title="Ghana Contracts Classifier API", version="1.0.0")

class ContractText(BaseModel):
    text: str

model = joblib.load("/mnt/data/models/tfidf_logreg_pipeline.pkl")

def clean_text(text: str) -> str:
    text = text.replace("_x000D_\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.post("/classify")
def classify(contract: ContractText):
    processed = clean_text(contract.text)
    pred = model.predict([processed])[0]
    probs = model.predict_proba([processed])[0]
    classes = model.classes_.tolist()
    return {"contract_type": pred, "all_probabilities": dict(zip(classes, probs.tolist()))}
