import os
import json
import pandas as pd
import random
from flask import Flask, request, render_template_string, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

try:
    from openai import OpenAI
except:
    OpenAI = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from datasets import load_dataset
    import torch
except:
    AutoModelForCausalLM = None

class LegalChatBot:
    INTENTS = ["Definition", "Procedure", "Eligibility", "Consequence", "DocumentRequirement", "RightsObligations", "Other"]

    def __init__(self, df, num_clusters=8, llm_model="deepseek/deepseek-chat"):
        self.df = df.copy()
        self.df["question"] = self.df["question"].astype(str).str.strip()
        self.df["answer"] = self.df["answer"].astype(str).str.strip()
        self.num_clusters = num_clusters
        self.llm_model = llm_model
 
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        questions = self.df["question"].tolist()
        embeddings = self.embedder.encode(questions, show_progress_bar=True)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
        self.df["cluster"] = self.kmeans.labels_
 
        self.openrouter_key = os.environ.get("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")
        self.client = None
        self.local_pipe = None

        if OpenAI is not None and self.openrouter_key:
            self.client = OpenAI(api_key=self.openrouter_key, base_url="https://openrouter.ai/api/v1") 
        else:
            print("[WARN] No LLM available!")

    def predict_cluster(self, user_question):
        emb = self.embedder.encode([user_question])
        return self.kmeans.predict(emb)[0]

    def get_few_shot_from_cluster(self, cluster_id, k=5):
        subset = self.df[self.df["cluster"] == cluster_id]
        samples = subset.sample(min(k, len(subset))).to_dict(orient="records")
        return [(s["question"], s.get("answer", "")) for s in samples]
 
    def build_prompt(self, fewshots, user_question):
        shots_text = ""
        for q,a in fewshots:
            shots_text += f"Example Question: {q}\nExample Answer: {a}\n---\n"
        prompt = f"""
You are an expert legal assistant.

INSTRUCTIONS:
1) Identify the user's intent (one of {self.INTENTS})
2) Suggest the best category (short phrase e.g., "Contract Law", "Family Law", "Criminal Law")
3) If the question is ambiguous or lacks jurisdiction/essential details, DO NOT guess. Ask a concise clarifying question and stop.
4) Otherwise, provide a concise, user-friendly answer (2-6 sentences). 
5) Include confidence: High/Medium/Low

FEW-SHOT EXAMPLES:
{shots_text}

USER QUESTION:
{user_question}

OUTPUT EXACTLY in JSON:
{{
  "intent": "",
  "category": "",
  "clarifying_question": "",
  "answer": "",
  "confidence": ""
}}
"""
        return prompt 


    def call_llm(self, prompt, max_tokens=400, temperature=0.1):
        if self.client:
            resp = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            choice = resp.choices[0]
            if hasattr(choice,"message") and hasattr(choice.message,"content"):
                return choice.message.content
            if hasattr(choice,"text"):
                return choice.text
            return str(resp)
        else:
            return json.dumps({
                "intent":"Other",
                "category":"Unknown",
                "clarifying_question":"No LLM available",
                "answer":"",
                "confidence":"Low"
            })

    def parse_model_output(self, text):
        try:
            start = text.find("{")
            end = text.rfind("}")+1
            jsontext = text[start:end]
            return json.loads(jsontext)
        except:
            return {
                "intent":"Other",
                "category":"Unknown",
                "clarifying_question":"Could not parse LLM output",
                "answer":"",
                "confidence":"Low"
            }

    def answer_question(self, user_question, k_shot=5):
        cid = self.predict_cluster(user_question)
        fewshots = self.get_few_shot_from_cluster(cid, k=k_shot)
        prompt = self.build_prompt(fewshots, user_question)
        raw = self.call_llm(prompt)
        parsed = self.parse_model_output(raw)
        parsed["cluster_id"] = cid
        parsed["prompt"] = prompt
        return parsed


def start_flask(df):
    bot = LegalChatBot(df, num_clusters=8)
    app = Flask(__name__)

    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Legal ChatBot</title>
        <style>
            body { font-family: Arial; margin: 20px; }
            textarea { width: 100%; height: 80px; }
            #answer { margin-top: 20px; white-space: pre-wrap; border:1px solid #ccc; padding:10px;}
        </style>
    </head>
    <body>
        <h2>Legal ChatBot</h2>
        <textarea id="question" placeholder="Type your legal question here..."></textarea><br>
        <button onclick="ask()">Ask</button>
        <div id="answer"></div>
        <script>
        async function ask(){
            const q = document.getElementById("question").value;
            const res = await fetch("/ask",{
                method:"POST",
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({question:q})
            });
            const data = await res.json();
            let output = "Cluster: "+data.cluster_id+"\\n";
            if(data.clarifying_question) {
                output += "Clarifying question: "+data.clarifying_question;
            } else {
                output += "Intent: "+data.intent+"\\nCategory: "+data.category+"\\nConfidence: "+data.confidence+"\\nAnswer: "+data.answer;
            }
            document.getElementById("answer").innerText = output;
        }
        </script>
    </body>
    </html>
    """

    @app.route("/")
    def home():
        return render_template_string(HTML_TEMPLATE)

    @app.route("/ask", methods=["POST"])
    def ask():
        data = request.json
        q = data.get("question","")
        if not q:
            return jsonify({"error":"question required"}), 400
        res = bot.answer_question(q)
        if "cluster_id" in res:
            res["cluster_id"] = int(res["cluster_id"])
        return jsonify(res)

    print("Web UI running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000)

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    dataset = load_dataset("dzunggg/legal-qa-v1")
    train = dataset["train"]
    df = pd.DataFrame(train)
    start_flask(df)
