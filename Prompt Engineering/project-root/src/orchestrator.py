import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from prompts import LANGUAGE_DETECTION_PROMPT, INTENT_CLASSIFICATION_PROMPT, DATE_INTERPRETATION_PROMPT, BOOKING_SLOT_FILLING_PROMPT
from prompts import BOOKING_CONFIRMATION_PROMPT,BOOKING_TOOL_TRIGGER_PROMPT, CANCEL_TOOL_TRIGGER_PROMPT, INFO_TOOL_TRIGGER_PROMPT, PASSENGER_VALIDATION_PROMPT
from prompts import DATE_INTERPRETATION_PROMPT

from .manager import ChatManager
from .tools import Tools

try:
    import jdatetime
    JALALI_AVAILABLE = True
except Exception:
    JALALI_AVAILABLE = False

logger = logging.getLogger("orchestrator")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def normalize_persian_digits(s: str) -> str:
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    return s.translate(trans)



class Orchestrator:

    def __init__(
        self,
        max_tool_rounds: int = 4,
    ):
        self.chat_manager = ChatManager()
        self.tools = Tools()
        self.max_tool_rounds = max_tool_rounds

     
    def _handle_booking(self, user_id: str, message: str) -> Dict[str, Any]:

        slot_resp = self.chat_manager.chat(
            user_id,
             BOOKING_SLOT_FILLING_PROMPT + DATE_INTERPRETATION_PROMPT + "\nUser message:\n" + message
        )
        response = slot_resp.strip()
        try:
            response = json.loads(slot_resp)  
        except json.JSONDecodeError: 
            return {"response": "Error parsing booking response."}
        
        if response.get('question'):
            return {"response": response['question']}
        
        slots=response.get('slots')
        validation_resp = self.chat_manager.chat(
            user_id,
            PASSENGER_VALIDATION_PROMPT
            + "\nBooking data:\n"
            + json.dumps(slots, ensure_ascii=False)
        )
        response = validation_resp.strip()
        try:
            response = json.loads(slot_resp)  
        except json.JSONDecodeError: 
            return {"response": "Error parsing booking response."}
        
        if response.get('question'):
            return {"response": response['question']}
        
        slots=response.get('slots')
        confirm_resp = self.chat_manager.chat(
            user_id,
            BOOKING_CONFIRMATION_PROMPT
            + "\nBooking data:\n"
            + json.dumps(slots, ensure_ascii=False)
        )
        if "تایید" not in confirm_resp and "confirm" not in confirm_resp.lower():
            return {"response": confirm_resp}

        tool_trigger = self.chat_manager.chat(
            user_id,
            BOOKING_TOOL_TRIGGER_PROMPT
            + "\nBooking data:\n"
            + json.dumps(slots, ensure_ascii=False)
        )

        tool_call = json.loads(tool_trigger)

        result = self.tools.run(
            tool_call["tool"],
            tool_call["params"]
        )

        return {
            "response": "رزرو شما با موفقیت انجام شد ",
            "tool_result": result
        }
 
    def _handle_cancel(self, user_id: str, message: str) -> Dict[str, Any]:
        tool_prompt = CANCEL_TOOL_TRIGGER_PROMPT + "\nUser message:\n" + message
        resp = self.chat_manager.chat(user_id, tool_prompt)

        if not resp.strip().startswith("{"):
            return {"response": resp}

        tool_call = json.loads(resp)
        result = self.tools.run(tool_call["tool"], tool_call["params"])

        return {
            "response": "بلیط با موفقیت کنسل شد ",
            "tool_result": result
        }
 
    def _handle_info(self, user_id: str, message: str) -> Dict[str, Any]: 
        tool_prompt = INFO_TOOL_TRIGGER_PROMPT + "\nUser message:\n" + message
        resp = self.chat_manager.chat(user_id, tool_prompt)

        if not resp.strip().startswith("{"):
            return {"response": resp}

        tool_call = json.loads(resp)
        result = self.tools.run(tool_call["tool"], tool_call["params"])

        return {
            "response": json.dumps(result, ensure_ascii=False)
        }
 
    def _handle_travel_suggestion(self, user_id: str, message: str) -> Dict[str, Any]:
        result = self.tools.run(
            "web_search",
            {"query": message}
        )
        return {
            "response": json.dumps(result, ensure_ascii=False)
        }
    
    def _interpret_date_with_llm(self, user_id: str, date_text: str) -> dict:

        prompt = DATE_INTERPRETATION_PROMPT + "\n\nUser message:\n" + date_text
        response_text = self.chat_manager.chat(user_id, prompt)
    
        try:
            date_info = json.loads(response_text)
        except Exception:
            date_info = {"jalali": None, "gregorian": None, "error": "llm_parse_failed"}
    
        return date_info
     

    def _detect_intent_llm(self, user_id: str, message: str) -> str:
        prompt = INTENT_CLASSIFICATION_PROMPT + "\n\nUser message:\n" + message
        try:
            intent_resp = self.chat_manager.chat(user_id, prompt).strip() 
            valid_intents = ["book_ticket", "cancel_ticket", "get_ticket_info", "travel_suggestion","rag_question"]
            if intent_resp in valid_intents:
                return intent_resp
            else:
                response = intent_resp.strip()
                try:
                    response = json.loads(intent_resp)  
                except Exception:
                    return "travel_suggestion" 
        
                if response.get('intent'):
                    return response.get('intent') 
        except Exception:
            return "travel_suggestion"
        

    def _detect_language_llm(self, user_id: str, message: str) -> str: 
        prompt = LANGUAGE_DETECTION_PROMPT + "\n\nUser message:\n" + message
        try:
            lang_resp = self.chat_manager.chat(user_id, prompt).strip().lower()
            if lang_resp in ["fa", "en"]:
                return lang_resp
            return "fa"   
        except Exception as e:
            print(str(e))
            return "fa"
        
 

    def _handle_rag(self, user_id: str, message: str) -> dict:
        rag_docs = self.tools.run(
            "rag_query",
            {"query": message}
        )
        context = "\n".join(
                f"{d['metadata'].get('source', '')}: {d['document']}" for d in rag_docs
            )
    
        prompt = ( 
            "\n\nCompany Knowledge:\n" +
            context +
            "\n\nUser question:\n" +
            message
        )
    
        answer = self.chat_manager.chat(user_id, prompt)
    
        return {
            "response": answer,
            "sources": rag_docs,
            "intent": "rag_question"
        }
 

    def run(self, user_id: str, user_message: str) -> dict: 
        lang = self._detect_language_llm(user_id, user_message)
        intent = self._detect_intent_llm(user_id, user_message)
    
        logger.info(f"User intent={intent} lang={lang}")
    
        if intent == "book_ticket":
            return self._handle_booking(user_id, user_message)
    
        if intent == "cancel_ticket":
            return self._handle_cancel(user_id, user_message)
    
        if intent == "get_ticket_info":
            return self._handle_info(user_id, user_message)
    
        if intent == "travel_suggestion":
            return self._handle_travel_suggestion(user_id, user_message)
    
        if intent == "rag_question":
            return self._handle_rag(user_id, user_message)
     
        assistant_text = self.chat_manager.chat(user_id, user_message)
    
        return {
            "response": assistant_text,
            "lang": lang,
            "intent": intent
        }