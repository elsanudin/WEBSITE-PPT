from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import random
from collections import deque
import re
from llama_cpp import Llama

app = Flask(__name__)
CORS(app)

# Path model
MODEL_PATH = r"c:\Users\Lenovo\AppData\Local\nomic.ai\GPT4All\Llama-3.2-3B-Instruct-Q4_0.gguf"

# Pastikan file model ada
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan di: {MODEL_PATH}")

# Load model
print("Memulai load model... (bisa makan waktu beberapa detik/menit tergantung model)")
llm = Llama(model_path=MODEL_PATH, n_ctx=512)  # pastikan context sesuai model
print("Model siap!")

# === Filter Prompt 18+ dan Custom Response ===
BLOCKED_KEYWORDS = ["sex", "porn", "bokep", "ngentot", "fuck", "18+", "masturbasi"]

CUSTOM_PATTERNS = {
    r"(?=.*\bnama\b)(?=.*\bkamu\b)(?=.*\bsiapa\b)": [
        "Saya Hexagon, ada yang bisa saya bantu?",
        "Hexagon di sini, siap menjawab pertanyaanmu!",
    ]
}

# Memory percakapan
conversation_history = deque(maxlen=5)  # batasi supaya tidak terlalu panjang


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400

    prompt = data["prompt"]

    # Filter 18+
    if any(word.lower() in prompt.lower() for word in BLOCKED_KEYWORDS):
        return jsonify({"response": "Maaf, saya tidak bisa menjawab pertanyaan itu."})

    # Custom response
    for pattern, responses in CUSTOM_PATTERNS.items():
        if re.search(pattern, prompt.lower()):
            reply = random.choice(responses)
            conversation_history.append({"user": prompt, "ai": reply})
            return jsonify({"response": reply})

    # Gabungkan history
    history_text = ""
    for c in conversation_history:
        history_text += f"User: {c['user']}\nAI: {c['ai']}\n"

    # Prompt baru ditambah history
    full_prompt = f"{history_text}User: {prompt}\nAI:"

    # Generate pakai llama_cpp
    try:
        response = llm(
            full_prompt,
            max_tokens=500,        # jangan terlalu tinggi
            temperature=0.3,
            top_p=0.9,
            repeat_penalty=1.2,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 800

    text = response["choices"][0]["text"].strip()
    conversation_history.append({"user": prompt, "ai": text})

    return jsonify({"response": text})


@app.route("/")
def index():
    return "Llama.cpp Flask API berjalan. Endpoint: POST /chat"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
