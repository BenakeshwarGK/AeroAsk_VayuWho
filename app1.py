import os
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from PIL import Image
from accelerate import infer_auto_device_map, dispatch_model
from transformers import AutoConfig

# Set Hugging Face transformers cache path to a writable directory
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache\transformers"


# Initialize Flask app
app = Flask(__name__)

# Load your classification model
model_path = "aeroclassify_model.h5"
classifier_model = load_model(model_path)

# Load label mappings
labels = [
    'A10', 'A400M', 'AG600', 'AH64', 'AV8B', 'An124', 'An22', 'An225', 'An72',
    'B1', 'B2', 'B21', 'B52', 'Be200', 'C130', 'C17', 'C2', 'C390', 'C5',
    'CH47', 'CL415', 'E2', 'E7', 'EF2000', 'EMB314', 'F117', 'F14', 'F15',
    'F16', 'F18', 'F22', 'F35', 'F4', 'H6', 'Il76', 'J10', 'J20', 'J35',
    'JAS39', 'JF17', 'JH7', 'KAAN', 'KC135', 'KF21', 'KJ600', 'Ka27', 'Ka52',
    'MQ9', 'Mi24', 'Mi26', 'Mi28', 'Mi8', 'Mig29', 'Mig31', 'Mirage2000',
    'P3', 'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25', 'Su34', 'Su57', 'TB001',
    'TB2', 'Tornado', 'Tu160', 'Tu22M', 'Tu95', 'U2', 'UH60', 'US2', 'V22',
    'V280', 'Vulcan', 'WZ7', 'XB70', 'Y20', 'YF23', 'Z10', 'Z19'
]

# Load Q&A CSV into a dictionary grouped by aircraft_id
qa_df = pd.read_csv("aircraft_qa_full.csv")
aircraft_qa = {}
for aircraft_id, group in qa_df.groupby('aircraft_id'):
    aircraft_qa[aircraft_id] = list(zip(group['question'], group['answer']))

# Load Gemma 2B model and tokenizer correctly with cache
model_id = "google/gemma-2b-it"
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load config
config = AutoConfig.from_pretrained(model_id)

# Initialize empty model
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

# Infer device map with max_memory set to 0 for CPU
device_map = infer_auto_device_map(model, max_memory={"cpu": "32GiB"}, no_split_module_classes=["GemmaDecoderLayer"])

# Load model with disk offload
chat_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device_map,
    offload_folder="offload",
    offload_state_dict=True,
    torch_dtype=torch.float16
)

# Pipeline for generation
generator = pipeline("text-generation", model=chat_model, tokenizer=tokenizer, device_map="auto")

# Helper to preprocess and predict aircraft from image
def predict_aircraft(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = classifier_model.predict(img_array)
    predicted_label = labels[np.argmax(prediction)]
    return predicted_label

# Helper to generate LLM response with Q&A context
def get_llm_response(aircraft_id, user_query):
    context = ""
    if aircraft_id in aircraft_qa:
        for q, a in aircraft_qa[aircraft_id]:
            context += f"Q: {q}\nA: {a}\n"
    prompt = f"""You are an aircraft expert chatbot.

Context:
{context}

User Question: {user_query}
Answer:"""

    result = generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
    response = result[0]['generated_text'].split("Answer:")[-1].strip()
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'question' not in request.form:
        return jsonify({'error': 'Missing image or question'}), 400

    img_file = request.files['image']
    user_question = request.form['question']

    if img_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    img_path = os.path.join("uploads", img_file.filename)
    os.makedirs("uploads", exist_ok=True)
    img_file.save(img_path)

    predicted_aircraft = predict_aircraft(img_path)
    answer = get_llm_response(predicted_aircraft, user_question)

    return render_template('result.html', aircraft=predicted_aircraft, answer=answer)



if __name__ == '__main__':
    app.run(debug=True)
