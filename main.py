from fastapi import FastAPI, Request, UploadFile, File,Body
from fastapi.middleware.cors import CORSMiddleware
from mongoDb import voice_collection
from mongoDb import save_meeting, save_voice
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
import pickle
import nltk
import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer

app = FastAPI()
load_dotenv()
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

googleclient = genai.Client(api_key = os.getenv("VITE_GOOGLE_GENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/meeting")
async def add_meeting(request: Request):
    data = await request.json()
    meeting_id = save_meeting(data)
    return {"meeting_id": str(meeting_id)}

@app.delete("/meeting/{meeting_id}")
async def delete_meeting(meeting_id: str):
    deleted_count = delete_meeting(meeting_id)
    if deleted_count:
        return {"detail": "Meeting deleted successfully"}
    return {"detail": "Meeting not found"}, 404

uri = os.getenv("VITE_MONGODB_URI")

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

class IntentClassifier:
    def __init__(self, model_path='chatbot_model.h5', words_path='words.pkl', classes_path='classes.pkl'):

        # Load trained model and preprocessed data
        self.model = tf.keras.models.load_model(model_path)
        self.words = pickle.load(open(words_path, 'rb'))
        self.classes = pickle.load(open(classes_path, 'rb'))
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.ignore_letters = ['?', '!', '.', ',', ';', ':']
        
        # Confidence threshold
        self.confidence_threshold = 0.5
    
    def clean_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence.lower())
        sentence_words = [
            self.lemmatizer.lemmatize(word) 
            for word in sentence_words 
            if word not in self.ignore_letters
        ]
        return sentence_words
    
    def create_bag_of_words(self, sentence):
        sentence_words = self.clean_sentence(sentence)
        bag = [0] * len(self.words)
        
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        
        return np.array(bag)
    
    def get_intent(self, sentence):
        # Create bag of words
        bow = self.create_bag_of_words(sentence)
        
        # Get prediction
        prediction = self.model.predict(np.array([bow]), verbose=0)[0]
        
        # Get the class with highest probability
        max_index = np.argmax(prediction)
        predicted_intent = self.classes[max_index]
        confidence = prediction[max_index]
        
        # If confidence is low, default to diagnosis
        if confidence < self.confidence_threshold:
            return "diagnosis"
        
        return predicted_intent

# Initialize the classifier (do this once)
classifier = IntentClassifier()

def classify_intent(sentence):
    return classifier.get_intent(sentence)

@app.post("/detect_intent/")
async def detect_intent(text: Text):
    intent = classify_intent(text.text)
    return JSONResponse(content={"intent": intent})