import os
import random
import re
import csv
import pandas as pd
import streamlit as st
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from rouge_score import rouge_scorer
import sacrebleu
from openai import OpenAI

class OpenRouterLLM:
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3-8b-instruct"):
        self.client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        self.model = model

    def chat(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {e}"

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"^\s*[qa]:\s*", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

class LegalChatbot:

    def __init__(self, openrouter_api_key: str, num_clusters: int = 8):
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY required")

        self.llm = OpenRouterLLM(api_key=openrouter_api_key)
        ds = load_dataset("dzunggg/legal-qa-v1")
         
        df = pd.DataFrame(ds["train"]) 
        df = df.dropna(subset=["question", "answer"])   
        df = df[(df["question"].str.strip() != "") & (df["answer"].str.strip() != "")] 
        df = df.reset_index(drop=True)
        
        
        self.train_df, self.test_df = train_test_split(df, test_size=0.2, random_state=42)
        self.train_df = self.train_df.reset_index(drop=True)
        self.test_df = self.test_df.reset_index(drop=True)
        self.df=self.train_df
        self.df["question"] = self.train_df["question"].astype(str).str.strip().apply(preprocess)
        self.df["answer"] = self.train_df["answer"].astype(str).str.strip().apply(preprocess)
        
 
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        questions = self.df[self.df["question"] != ""]["question"].tolist()
        embeddings = self.embedder.encode(questions, show_progress_bar=True) 
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
        self.df["cluster"] = self.kmeans.labels_.astype(int)

 
        self.cluster_names = self._name_clusters()
 

    def _name_clusters(self):
        names = {}
        for c in range(self.num_clusters):
            idxs =  self.df.index[self.df["cluster"] == c].tolist()
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
        
        prompt = f"""Is the following legal question ambiguous or 
                   lacking essential details so a clarifying question is required? 
                   Answer only "yes" or "no".\n\nQuestion: {question}"""
        try:
            r = self.llm.chat(prompt, max_tokens=10, temperature=0.0).lower()
            return "yes" in r
        except Exception:
            return False
        
    def find_similarity(self, str1:str, str2:str) -> bool:
        
        prompt = f"""Are the following two responses similar in meaning?
                    Determine if the two responses convey the same meaning or intent 
                    Answer only "yes" or "no".

                    Response 1: {str1}
                    Response 2: {str2}"""
        try:
            r = self.llm.chat(prompt, max_tokens=10, temperature=0.0).lower()
            return "yes" in r
        except Exception:
            return False

    def get_clarifying_question(self, question: str) -> str:
        prompt = f"""The following legal question may be ambiguous or lack details. 
                   Generate ONE concise clarifying question (one sentence) that 
                   would let you give a precise legal answer.\n\nQuestion: {question}"""
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
            f"You are an expert legal assistant.\n"
            f"Category: {self.cluster_names.get(cluster_id, 'General')}\n\n"
            f"Examples:\n{examples}\n\n"
            f"Now answer this user question concisely in 2-3 sentences, including only the key legal points. "
            f"Do not provide step-by-step instructions or extra explanations. Answer in a direct and factual style, similar to the examples provided. "
            f"If the answer depends on jurisdiction, mention it.\n\n"
            f"User question: {user_question}\n\n"
            f"Answer:")
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

    def evaluate(self, n_samples: int = 10, similarity_threshold: float = 0.75):
        if self.test_df.empty:
            return "No test data available"
    
        results = []
        sample_df = self.test_df[self.test_df["question"] != ""].sample(min(n_samples, len(self.test_df)))
        sample_df["question"] = sample_df["question"].astype(str).apply(preprocess)
        sample_df["answer"] = sample_df["answer"].astype(str).apply(preprocess)
    
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
        for _, row in sample_df.iterrows():
            question, true_answer = row["question"], row["answer"]
            cluster_id, _ = self.detect_cluster(question)
            prompt = self.build_prompt_for_answer(question, cluster_id)
            pred_answer = self.llm.chat(prompt, max_tokens=200) 
            bleu = sacrebleu.sentence_bleu(pred_answer, [true_answer]).score
            rouge_scores = scorer.score(true_answer, pred_answer)
            rouge1 = rouge_scores['rouge1'].fmeasure
            rougeL = rouge_scores['rougeL'].fmeasure

            is_correct = self.find_similarity(true_answer,pred_answer)
            results.append({
                "question": question,
                "true_answer": true_answer,
                "pred_answer": pred_answer,
                "bleu": bleu,
                "rouge1": rouge1,
                "rougeL": rougeL,
                "is_correct": is_correct
            })
    
        avg_bleu = sum(r["bleu"] for r in results)/len(results)
        avg_rouge1 = sum(r["rouge1"] for r in results)/len(results)
        avg_rougeL = sum(r["rougeL"] for r in results)/len(results)
        accuracy = sum(r["is_correct"] for r in results)/len(results)
    
        return results, avg_bleu, avg_rouge1, avg_rougeL, accuracy
    
    def save_feedback(self, question: str, answer: str, rating: int = None, like: int = None, comment: str = ""):
        file = "feedback.csv"
        row = [datetime.now(datetime.timezone.utc).isoformat(), question, answer, rating if rating is not None else "", like if like is not None else "", comment]
        write_header = not os.path.exists(file)
        with open(file, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["timestamp_utc", "question", "answer", "rating_1_5", "like_1_or_0", "comment"])
            w.writerow(row)


st.set_page_config(page_title="Legal Assistant Chatbot", layout="centered")
st.title("Legal Assistant Chatbot")


import streamlit as st
import os
 
OPENROUTER_API_KEY = "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"
 
if "bot" not in st.session_state:
    with st.spinner("Creating chatbot (this may take a minute)..."):
        st.session_state.bot = LegalChatbot(openrouter_api_key=OPENROUTER_API_KEY, num_clusters=8)
        st.success("Chatbot created!")
 
if "last_result" not in st.session_state:
    st.session_state.last_result = None
 
category = st.selectbox("Choose category", ["Auto-detect", "Civil", "Criminal", "Family", "Contract", "Tax"])
 
question = st.text_area("Write your legal question:", height=120)

if st.button("Ask"):
    if question.strip() == "":
        st.warning("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            chosen_category = None if category == "Auto-detect" else category
            response = st.session_state.bot.answer(question, chosen_category=chosen_category)
            st.session_state.last_result = {"question": question, "response": response}
 
if st.session_state.last_result:
    data = st.session_state.last_result["response"]

    if data.get("ambiguous"):
        st.warning("Clarifying question required.")
        st.write("### Clarifying Question")
        st.write(data["clarifying_question"])
        clar_text = st.text_area("Your clarification:")

        if st.button("Send clarification"):
            if clar_text.strip():
                with st.spinner("Analyzing with your clarification..."):
                    response = st.session_state.bot.answer(
                        st.session_state.last_result["question"], 
                        clarification_answer=clar_text
                    )
                    st.session_state.last_result["response"] = response
            else:
                st.error("Please write a clarification message.")
    else:
        st.success("Answer received")
        st.write(f"### Category: {data.get('category_name','')}")
        st.write("### Answer:")
        st.write(data.get("answer",""))

        st.markdown("---")
        st.write("### Feedback")
        rating = st.selectbox("Rating (1–5)", ["-", 1, 2, 3, 4, 5])
        like = st.radio("Reaction:", ["None", "Like", "Dislike"])
        comment = st.text_area("Comment:")

        if st.button("Submit feedback"):
            fb_payload = {
                "question": st.session_state.last_result["question"],
                "answer": data.get("answer",""),
                "rating": None if rating == "-" else int(rating),
                "like": 1 if like == "Like" else 0 if like == "Dislike" else None,
                "comment": comment,
            }
            try:
                st.session_state.bot.save_feedback(**fb_payload)
                st.success("Thank you for your feedback!")
            except Exception as e:
                st.error(f"Failed to save feedback: {e}")
