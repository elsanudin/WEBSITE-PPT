from flask import Flask, request, jsonify
from flask_cors import CORS
from gpt4all import GPT4All
import os
import random
from collections import deque
import re

app = Flask(__name__)
CORS(app)

MODEL_PATH = r"c:\Users\Lenovo\gpt4all\resources\Llama-3.2-3B-Instruct-Q4_K_L.gguf"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model tidak ditemukan di: {MODEL_PATH}")

print("Memulai load model... (bisa agak lama)")
gpt = GPT4All(MODEL_PATH, device="cpu")   
print("Model siap!")

BLOCKED_KEYWORDS = ["sex", "porn", "bokep", "ngentot", "fuck", "18+", "masturbasi"]

CUSTOM_PATTERNS = {
    r"(?=.*\bh[ao]{1,2}l+o+\b)(?=.*\bai\b)(?=.*\b(nama|namu|nm|na)\b)(?=.*\b(kamu|kmu|km)\b)(?=.*\b(siapa|sypa|sipa|spa)\b)": [
        "Saya AI Hexagon, ada yang bisa saya bantu?",
        "Hexagon di sini, siap bantu kamu!",
        "Halo, saya Hexagon. Apa yang bisa saya lakukan untukmu?",
    ],
    r"(?=.*\b(nama|namu|nm|na)\b)(?=.*\b(kamu|kmu|km)\b)(?=.*\b(siapa|sypa|sipa|spa)\b)": [
        "Hexagon, ada yang bisa saya bantu?",
        "Saya Hexagon, siap menjawab pertanyaanmu!",
        "Hexagon di sini, apa yang bisa saya bantu?",
    ],
    r"(?=.*\b(nama|namu|nm|na)\b)(?=.*\b(kamu|kmu|km)\b)": [
        "Saya Hexagon, siap membantu kamu!",
        "Hexagon di sini, apa yang bisa saya bantu?",
    ],
    r"(?=.*\bh[ao]{1,2}l+o+\b)(?=.*\bai\b)(?=.*\b(kamu|nama|siapa|namamu)\b)": [
        "Saya AI Hexagon, ada yang bisa saya bantu?",
        "Hexagon di sini, siap bantu kamu!",
        "Halo, saya Hexagon. Apa yang bisa saya lakukan untukmu?",
    ],
    r"(?=.*\b(siapa|sypa|sipa)\b)(?=.*\b(kamu|namamu|nama)\b)": [
        "Hexagon, ada yang bisa saya bantu?",
        "Saya Hexagon, siap menjawab pertanyaanmu!",
        "Saya AI Hexagon, siap membantu kamu!",
        "Hexagon di sini, apa yang bisa saya bantu?",
    ],
    r"(?=.*\b(siapa|sypa|sipa|spa)\b)(?=.*\b(kamu|kmu|km)\b)": [
        "Saya Hexagon, siap membantu kamu!",
        "Hexagon di sini, apa yang bisa saya bantu?",
    ],
}

conversation_history = deque(maxlen=10)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    prompt = data["prompt"]

    max_tokens = data.get("max_tokens", 1050)
    temp = data.get("temp", 0.3)
    top_p = data.get("top_p", 0.9)
    repeat_penalty = data.get("repeat_penalty", 1.2)

    if any(word.lower() in prompt.lower() for word in BLOCKED_KEYWORDS):
        return jsonify({"response": "Maaf, saya tidak bisa menjawab pertanyaan itu."})

    for pattern, responses in CUSTOM_PATTERNS.items():
        if re.search(pattern, prompt.lower()):
            reply = random.choice(responses)
            conversation_history.append({"user": prompt, "ai": reply})
            return jsonify({"response": reply})

    try:
        history_text = ""
        if conversation_history:
            last_conv = conversation_history[-1]
            history_text = f"ini topic pembahasan sebelumnya {last_conv['user']} nggak usah di jawab lagi kamu fokus di jawaban terbaru \n"

        full_prompt = f"{history_text}\n ini prompt terbaru {prompt}"

        resp = gpt.generate(
            full_prompt,
            max_tokens=max_tokens,
            temp=temp,
            top_p=top_p,
            repeat_penalty=repeat_penalty
        )

        text = resp.strip()
        conversation_history.append({"user": prompt})

        return jsonify({"response": text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return "GPT4All Flask API berjalan. Endpoint: POST /chat"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
