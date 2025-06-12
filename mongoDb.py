from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("VITE_MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

db = client["health_chat"]

voice_collection = db["voice"]

def save_voice(voice_data):
    result = voice_collection.insert_one(voice_data)
    return result.inserted_id

test_collection = db["test"]
booking_collection = db["booking"]
meeting_collection  = db["room_id"]

def save_test(test_data):
    result = test_collection.insert_one(test_data)
    return result.inserted_id

def save_booking(booking_data):
    result = booking_collection.insert_one(booking_data)
    return result.inserted_id

def save_meeting(meeting_data):
    result = meeting_collection.insert_one(meeting_data)
    return result.inserted_id
