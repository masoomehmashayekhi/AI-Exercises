import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from prompts import LANGUAGE_DETECTION_PROMPT, INTENT_CLASSIFICATION_PROMPT, DATE_INTERPRETATION_PROMPT


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
        max_tool_rounds: int = 3,
    ):
        self.chat_manager = ChatManager()
        self.max_tool_rounds = max_tool_rounds


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
            valid_intents = ["book_ticket", "cancel_ticket", "get_ticket_info", "travel_suggestion"]
            if intent_resp in valid_intents:
                return intent_resp
            return "travel_suggestion"   
        except Exception:
            return "travel_suggestion"
        

    def _detect_language_llm(self, user_id: str, message: str) -> str: 
        prompt = LANGUAGE_DETECTION_PROMPT + "\n\nUser message:\n" + message
        try:
            lang_resp = self.chat_manager.chat(user_id, prompt).strip().lower()
            if lang_resp in ["fa", "en"]:
                return lang_resp
            return "fa"   
        except Exception:
            return "fa"
        

    def _ask_model(self, user_id: str, prompt_text: str) -> str:
        try:
            return self.chat_manager.chat(user_id, prompt_text)
        except Exception as e:
            logger.exception("LLM call failed")
            return f"ERROR: model_call_failed: {str(e)}"


    def _extract_tool_call(self, assistant_text: str) -> Optional[Dict[str, Any]]:
        if not assistant_text or not assistant_text.strip():
            return None
        try:
            parsed = json.loads(assistant_text.strip())
            if isinstance(parsed, dict) and "tool" in parsed:
                return parsed
        except Exception:
            pass


        m = re.search(r"(\{.*\"tool\".*\})", assistant_text, flags=re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                if "tool" in parsed:
                    return parsed
            except Exception:
                pass

        return None


    def run(self, user_id: str, user_message: str) -> dict:
        lang = self._detect_language_llm(user_id, user_message)
        intent = self._detect_intent_llm(user_id, user_message)
    
        tools_used = []
        tool_results = []
        sources = []
    
        logger.info(f"User intent={intent} lang={lang}")
    
        tool_map = {
            "book_ticket": "api_book_ticket",
            "cancel_ticket": "api_cancel_ticket",
            "get_ticket_info": "api_get_ticket_info",
            "travel_suggestion": "web_search"
        }
    
        selected_tool = tool_map.get(intent, None)
        tool_rounds = 0  
    
        while selected_tool and tool_rounds < self.max_tool_rounds:
            params = {"message": user_message}
     
            if "date" in params and params["date"]:
                date_info = self._interpret_date_with_llm(user_id, params["date"])
                if date_info.get("gregorian"):
                    params["date"] = date_info["gregorian"]
                if date_info.get("jalali"):
                    params["_jalali_date"] = date_info["jalali"]
                if date_info.get("error"):
                    params["_date_parse_error"] = date_info["error"]
    
            try:
                tool_result = self.chat_manager.tools.run(selected_tool, params)
            except Exception as e:
                tool_result = {"error": f"tool_exception: {str(e)}"}
    
            tools_used.append(selected_tool)
            tool_results.append(tool_result)
    
            if isinstance(tool_result, dict):
                if "results" in tool_result:
                    sources.append(tool_result.get("results"))
                elif "data" in tool_result:
                    sources.append(tool_result.get("data"))
     
            follow_payload = {
                "original_user_message": user_message,
                "tool_call": {"tool": selected_tool, "params": params},
                "tool_result": tool_result
            }
            follow_prompt = json.dumps(follow_payload, ensure_ascii=False)
            assistant_text = self.chat_manager.chat(user_id, follow_prompt)
    
            tool_rounds += 1
    
            
            break   
    
        if not selected_tool: 
            assistant_text = self.chat_manager.chat(user_id, user_message)
    
        return {
            "response": assistant_text,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "sources": sources,
            "lang": lang,
            "intent": intent
        }