import os
import json
import random
import csv
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import requests

class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3-8b-instruct"):
        self.api_key = api_key
        self.model = model
        self.url = "https://openrouter.ai/api/v1/chat/completions"

    def chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        resp = requests.post(self.url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # safe access
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data)

class LegalChatbot:
    INTENTS = ["Definition", "Procedure", "Eligibility", "Consequence",
               "DocumentRequirement", "RightsObligations", "Other"]

    def __init__(self, openrouter_api_key: str, num_clusters: int = 8):
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        self.llm = OpenRouterLLM(openrouter_api_key) 
        ds = load_dataset("dzunggg/legal-qa-v1")
        
        split = "train" if "train" in ds else list(ds.keys())[0]
        self.df = ds[split].to_pandas()
        
        if "question" not in self.df.columns or "answer" not in self.df.columns:
            raise RuntimeError("Dataset must contain 'question' and 'answer' fields")

        self.df["question"] = self.df["question"].astype(str).str.strip()
        self.df["answer"] = self.df["answer"].astype(str).str.strip()

 
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        questions = self.df["question"].tolist()
        embeddings = self.embedder.encode(questions, show_progress_bar=True) 
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
        self.df["cluster"] = self.kmeans.labels_.astype(int)

 
        self.cluster_names = self._name_clusters()
 

    def _name_clusters(self):
        names = {}
        for c in range(self.num_clusters):
            idxs = [i for i, lab in enumerate(self.df["cluster"]) if lab == c]
            samples = random.sample(idxs, min(6, len(idxs)))
            sample_qs = [self.df.loc[i, "question"] for i in samples]
            prompt = (
                "You are an expert legal classifier. Given these example legal questions, "
                "provide a short category name (2-4 words) such as 'Contract Law' or 'Criminal Law'.\n\n"
                + "\n".join(f"- {q}" for q in sample_qs)
            )
            try:
                name_raw = self.llm.chat(prompt, max_tokens=20, temperature=0.2)
                
                name = name_raw.splitlines()[0].strip().strip('"').strip()
                if len(name) == 0:
                    name = f"Cluster {c}"
            except Exception:
                name = f"Cluster {c}"
            names[c] = name
        return names

    def detect_cluster(self, user_question: str):
        vec = self.embedder.encode([user_question])
        cid = int(self.kmeans.predict(vec)[0])
        return cid, self.cluster_names.get(cid, f"Cluster {cid}")

    def is_ambiguous(self, question: str) -> bool:
        
        prompt = f'Is the following legal question ambiguous or lacking essential details so a clarifying question is required? Answer only "yes" or "no".\n\nQuestion: "{question}"'
        try:
            r = self.llm.chat(prompt, max_tokens=10, temperature=0.0).lower()
            return "yes" in r
        except Exception:
            return False

    def get_clarifying_question(self, question: str) -> str:
        prompt = f'The following legal question may be ambiguous or lack details. Generate ONE concise clarifying question (one sentence) that would let you give a precise legal answer.\n\nQuestion: "{question}"'
        try:
            return self.llm.chat(prompt, max_tokens=60, temperature=0.2).strip()
        except Exception:
            return "Could you please provide more details (jurisdiction/date/type of contract/etc.)?"

    def build_prompt_for_answer(self, user_question: str, cluster_id: int, k_shot: int = 4) -> str:
        idxs = [i for i, lab in enumerate(self.df["cluster"]) if lab == cluster_id]
        if len(idxs) == 0:
            examples = ""
        else:
            chosen = random.sample(idxs, min(k_shot, len(idxs)))
            examples = "\n\n".join(f"Q: {self.df.loc[i,'question']}\nA: {self.df.loc[i,'answer']}" for i in chosen)

        prompt = (
            f"You are an expert legal assistant. Category: {self.cluster_names.get(cluster_id, 'General')}\n\n"
            f"Examples:\n{examples}\n\n"
            f"Now answer this user question concisely and clearly. If the answer depends on jurisdiction, say so.\n\nUser question: {user_question}\n\nAnswer:"
        )
        return prompt

    def answer(self, user_question: str, chosen_category: int = None, clarification_answer: str = None):


        if clarification_answer:
            user_question = user_question + " Clarification: " + clarification_answer


        if clarification_answer is None and self.is_ambiguous(user_question):
            clar_q = self.get_clarifying_question(user_question)
            return {
                "ambiguous": True,
                "clarifying_question": clar_q,
                "original_question": user_question
            }


        if chosen_category is not None:
            cluster_id = int(chosen_category)
            cluster_name = self.cluster_names.get(cluster_id, f"Cluster {cluster_id}")
        else:
            cluster_id, cluster_name = self.detect_cluster(user_question)

        prompt = self.build_prompt_for_answer(user_question, cluster_id)
        try:
            answer_text = self.llm.chat(prompt, max_tokens=400, temperature=0.1)
        except Exception as e:
            answer_text = f"LLM error: {e}"

        return {
            "ambiguous": False,
            "category_id": int(cluster_id),
            "category_name": cluster_name,
            "answer": answer_text
        }

    def save_feedback(self, question: str, answer: str, rating: int = None, like: int = None, comment: str = ""):
        file = "feedback.csv"
        row = [datetime.utcnow().isoformat(), question, answer, rating if rating is not None else "", like if like is not None else "", comment]
        write_header = not os.path.exists(file)
        with open(file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp_utc", "question", "answer", "rating_1_5", "like_1_or_0", "comment"])
            w.writerow(row)


app = Flask(__name__)
CHATBOT = None   

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Legal Chatbot</title>
  <style>
    body{font-family: Arial; max-width:900px; margin:30px;}
    textarea{width:100%; height:90px; padding:8px;}
    select{padding:6px;}
    .box{border:1px solid #ddd;padding:12px;border-radius:6px;margin-top:12px;}
    button{padding:8px 12px;margin:6px 0;}
    #clarify-block, #feedback-block{display:none; margin-top:12px;}
  </style>
</head>
<body>
  <h2>Legal Assistant Chatbot</h2>
  <div>
    <label>Choose category (optional): </label>
    <select id="category">
      <option value="">Auto-detect</option>
      {% for cid, cname in categories.items() %}
        <option value="{{cid}}">{{cname}}</option>
      {% endfor %}
    </select>
  </div>

  <div style="margin-top:8px;">
    <textarea id="question" placeholder="Type your legal question here..."></textarea>
    <button id="ask-btn">Ask</button>
  </div>

  <div id="response" class="box" style="display:none;"></div>

  <div id="clarify-block" class="box">
    <div><b>Clarifying question:</b></div>
    <div id="clarify-text" style="margin:8px 0;"></div>
    <textarea id="clarify-answer" placeholder="Write your answer to the clarifying question here..."></textarea><br>
    <button id="send-clarify">Send clarification</button>
  </div>

  <div id="feedback-block" class="box">
    <div><b>Give feedback</b></div>
    <label>Rating (1–5): </label>
    <select id="rating"><option value="">-</option>{% for i in range(1,6) %}<option value="{{i}}">{{i}}</option>{% endfor %}</select>
    &nbsp; <button id="like-btn">like</button> <button id="dislike-btn">dislike</button>
    <div style="margin-top:8px;">
      <textarea id="fb-comment" placeholder="Optional comment..."></textarea><br>
      <button id="submit-fb">Submit feedback</button>
    </div>
  </div>

<script>
let lastResult = null;
document.getElementById("ask-btn").onclick = async function(){
  const q = document.getElementById("question").value.trim();
  const cat = document.getElementById("category").value;
  if(!q){ alert("Please type a question."); return; }
  document.getElementById("response").style.display = "block";
  document.getElementById("response").innerText = "Thinking...";
  // call ask
  const res = await fetch("/ask", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({question: q, category: cat})
  });
  const data = await res.json();
  lastResult = {"question": q, "category": cat, "response": data};
  if(data.ambiguous){
    // show clarifying question block
    document.getElementById("response").innerText = "Clarifying question required.";
    document.getElementById("clarify-text").innerText = data.clarifying_question;
    document.getElementById("clarify-block").style.display = "block";
    document.getElementById("feedback-block").style.display = "none";
  } else {
    document.getElementById("clarify-block").style.display = "none";
    document.getElementById("response").innerHTML = "<b>Category:</b> " + data.category_name + "<br><br><b>Answer:</b><br>" + data.answer;
    document.getElementById("feedback-block").style.display = "block";
  }
};

document.getElementById("send-clarify").onclick = async function(){
  const clar = document.getElementById("clarify-answer").value.trim();
  if(!clar){ alert("Write a short clarification answer."); return; }
  // call ask again with clarification
  const payload = {
    question: lastResult.question,
    category: lastResult.category,
    clarification: clar
  };
  document.getElementById("response").innerText = "Thinking with your clarification...";
  const res = await fetch("/ask", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  const data = await res.json();
  lastResult = {"question": lastResult.question, "category": lastResult.category, "response": data};
  if(data.ambiguous){
    document.getElementById("response").innerText = "Still ambiguous: " + data.clarifying_question;
    document.getElementById("clarify-text").innerText = data.clarifying_question;
  } else {
    document.getElementById("response").innerHTML = "<b>Category:</b> " + data.category_name + "<br><br><b>Answer:</b><br>" + data.answer;
    document.getElementById("clarify-block").style.display = "none";
    document.getElementById("feedback-block").style.display = "block";
  }
};

document.getElementById("submit-fb").onclick = async function(){
  if(!lastResult) { alert("No response to give feedback on."); return; }
  const rating = document.getElementById("rating").value;
  const like = window._like_state || null;
  const comment = document.getElementById("fb-comment").value;
  const payload = {
    question: lastResult.question,
    answer: lastResult.response.answer || "",
    rating: rating ? parseInt(rating) : null,
    like: like,
    comment: comment || ""
  };
  await fetch("/feedback", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(payload)
  });
  alert("Thank you for feedback!");
  // reset
  document.getElementById("rating").value = "";
  document.getElementById("fb-comment").value = "";
  window._like_state = null;
};

document.getElementById("like-btn").onclick = function(){ window._like_state = 1; alert("Marked Like") };
document.getElementById("dislike-btn").onclick = function(){ window._like_state = 0; alert("Marked Dislike") };

</script>
</body>
</html>
"""

@app.route("/")
def index():
    cats = CHATBOT.cluster_names if CHATBOT else {}
    return render_template_string(HTML_TEMPLATE, categories=cats)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    q = data.get("question","").strip()
    cat = data.get("category","")
    clarification = data.get("clarification", None)

    if not q:
        return jsonify({"error":"question required"}), 400

    chosen = int(cat) if (cat is not None and cat != "") else None

    try:
        res = CHATBOT.answer(q, chosen_category=chosen, clarification_answer=clarification)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


    if "category_id" in res:
        res["category_id"] = int(res["category_id"])
    if "ambiguous" in res:
        res["ambiguous"] = bool(res["ambiguous"])

    return jsonify(res)

@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json() or {}
    question = data.get("question","")
    answer = data.get("answer","")
    rating = data.get("rating", None)
    like = data.get("like", None)
    comment = data.get("comment","")
    try:
        CHATBOT.save_feedback(question=question, answer=answer, rating=rating, like=like, comment=comment)
    except Exception as e:
        return jsonify({"status":"error","error":str(e)}), 500
    return jsonify({"status":"ok"})
 

if __name__ == "__main__":
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        key = input("Enter your OpenRouter API key: ").strip()
    CHATBOT = LegalChatbot(openrouter_api_key=key, num_clusters=8)
    print("Starting web server at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
