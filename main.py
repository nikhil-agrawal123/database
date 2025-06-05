from fastapi import FastAPI, UploadFile, File,Body
from fastapi.middleware.cors import CORSMiddleware
from mongoDb import voice_collection
from mongoDb import  save_voice
from pymongo import MongoClient
from fastapi.responses import JSONResponse
import gridfs
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
from bson import ObjectId
import speech_recognition as sr
from pydub import AudioSegment
import tempfile
from google import genai
from pydantic import BaseModel
from gtts import gTTS
import io
from googletrans import Translator

app = FastAPI()
load_dotenv()

googleclient = genai.Client(api_key = os.getenv("VITE_GOOGLE_GENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
uri = os.getenv("VITE_MONGO_URI")

client = MongoClient(uri)
db = client["health_chat"]
fs = gridfs.GridFS(db)

@app.post("/record")
async def record_voice(audio: UploadFile = File(...)):
    contents = await audio.read()
    file_id = fs.put(contents, filename=audio.filename, content_type=audio.content_type)

    # Save to a temp file for transcription
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(contents)
        temp_audio_path = temp_audio.name

    # Convert webm to wav for SpeechRecognition
    wav_path = temp_audio_path.replace(".webm", ".wav")
    try:
        sound = AudioSegment.from_file(temp_audio_path, format="webm")
        sound.export(wav_path, format="wav")

        # Transcribe using SpeechRecognition
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
            except Exception as e:
                text = f"Transcription failed: {str(e)}"
    except Exception as e:
        text = f"Audio conversion/transcription failed: {str(e)}"
    finally:
        # Clean up temp files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    # Save voice info and transcription using your custom function
    save_voice({
        "file_id": str(file_id),
        "filename": audio.filename,
        "content_type": audio.content_type,
        "transcription": text
    })

    return {"file_id": str(file_id), "transcription": text}

@app.get("/audio/{file_id}")
async def get_audio(file_id: str):
    file = fs.get(ObjectId(file_id))
    return StreamingResponse(file, media_type=file.content_type)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/gemini")
async def generate_response(request: PromptRequest):
    response = googleclient.models.generate_content(
        model="gemini-2.0-flash",
        contents="you are a professional medical assistant, conversing with a patient. initially the patient tells you their symptoms you have to follow up with question regarding duration and severity of the symptoms. ask a few follow up question to have a basic idea of the disease and dont ask for past history or any personal information. behave professionally and at the end generate a small report covering the symptoms duration severity and possible disease the patient is suffering from ask the questions one by one not collectivly and you have access to previous things the patient has said and according the the front line assess what the duration of the symptoms and other things are and dont repeat a question twice" + request.prompt,
    )
    return {"text": response.text}

@app.post("/tts")
async def text_to_speech(text: str = Body(...), language: str = Body("en")):
    # Generate MP3 using gTTS
    tts = gTTS(text,lang=language)
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    # Store in MongoDB
    result = voice_collection.insert_one({
        "audio": mp3_fp.getvalue(),
        "content_type": "audio/mpeg"
    })
    return {"file_id": str(result.inserted_id)}

@app.get("/tts/{file_id}")
async def get_tts_audio(file_id: str):
    doc = voice_collection.find_one({"_id": ObjectId(file_id)})
    if not doc:
        return {"error": "File not found"}
    return StreamingResponse(io.BytesIO(doc["audio"]), media_type=doc["content_type"])

class Text(BaseModel):
    text: str
    class Config:
        from_attributes = True

translator = Translator()

@app.post("/english/")
async def english(text: Text):
    translation =  translator.translate(text.text, dest='en')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/hindi/")
async def hindi(text: Text):
    translation =  translator.translate(text.text, src="en", dest='hi')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/punjabi/")
async def punjabi(text: Text):
    translation =  translator.translate(text.text, dest='pa')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/gujarati/")
async def gujarati(text: Text):
    translation =  translator.translate(text.text, dest='gu')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/bengali/")
async def bengali(text: Text):
    translation =  translator.translate(text.text, dest='bn')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/tamil/")
async def tamil(text: Text):
    translation =  translator.translate(text.text, dest='ta')
    return JSONResponse(content={"Translation": translation.text})

@app.post("/telugu/")
async def telugu(text: Text):
    translation =  translator.translate(text.text, dest='te')
    return JSONResponse(content={"Translation": translation.text})