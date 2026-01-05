import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer

MODEL_PATH = "models/vosk-model-small-en-us-0.15"

q = queue.Queue()
model = Model(MODEL_PATH)
samplerate = 16000

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(bytes(indata))

def offline_speech_to_text(duration=5):
    recognizer = KaldiRecognizer(model, samplerate)

    with sd.RawInputStream(samplerate=samplerate,
                           blocksize=8000,
                           dtype="int16",
                           channels=1,
                           callback=callback):
        print("🎙 Listening...")
        for _ in range(int(duration * samplerate / 8000)):
            data = q.get()
            if recognizer.AcceptWaveform(data):
                pass

    result = json.loads(recognizer.FinalResult())
    return result.get("text", "")










import streamlit as st
from speech import offline_speech_to_text

st.title("🏦 Banking Chatbot")

# Button to speak
if st.button("🎙 Speak"):
    with st.spinner("Listening..."):
        spoken_text = offline_speech_to_text()

    if spoken_text:
        st.success(f"You said: {spoken_text}")

        # 🔁 Send spoken text into chatbot
        user_input = spoken_text

        # TODO: replace this with your chatbot response
        response = f"Bank bot received: {user_input}"

        st.write("🤖 Bot:", response)
    else:
        st.warning("No speech detected")
