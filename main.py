from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
app = FastAPI()
client = genai.Client(api_key=API_KEY)

class TranslationRequest(BaseModel):
    text: str
    lang: str

@app.post("/translate")
async def translate(req: TranslationRequest):
    """
    Translate input text into a target language.
    Example: {"text": "Hello", "lang": "Hindi"}
    """
    prompt = f"Translate the following text to {req.lang}. Only return the translated text, nothing else:\n\n{req.text}"
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return {"translation": response.text}

@app.post("/caption")
async def caption(image: UploadFile, lang: str = Form(...)):
    """
    Generate image description (caption) in target language.
    Example: upload file + lang="Hindi"
    """
    # read file bytes
    img_bytes = await image.read()

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[
            f"Describe this image in {lang}.",
            {
                "inline_data": {
                    "mime_type": image.content_type,
                    "data": img_bytes
                }
            }
        ]
    )
    return {"caption": response.text}
