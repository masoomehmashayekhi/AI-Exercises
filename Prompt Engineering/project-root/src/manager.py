# src/manager.py
from typing import Dict, Any
from .tools import TravelTools
from prompts import SYSTEM_PROMPT
from datetime import datetime
import json
import openai


class ChatManager:

    def __init__(self, model_name: str = "meta-llama/llama-4-maverick-17b-128e-instruct"):
        self.tools = TravelTools()
        self.session_memory = {}   
        self.model = model_name    

    def _build_prompt(self, user_id: str, user_message: str) -> str:

        history = self.session_memory.get(user_id, [])

        conversation_block = ""
        for turn in history:
            conversation_block += f"User: {turn['user']}\n"
            conversation_block += f"Assistant: {turn['assistant']}\n"

        final_prompt = (
            f"{SYSTEM_PROMPT}\n"
            f"Conversation history:\n{conversation_block}\n"
            f"User message:\n{user_message}\n"
            f"Today’s datetime (Gregorian): {datetime.now()}\n"
            f"Today’s datetime (Jalali): <converted by system>\n"
        )

        return final_prompt

    def _detect_tool_call(self, response: str) -> Dict[str, Any]:
        try:
            data = json.loads(response)
            if "tool" in data:
                return data
            return None
        except:
            return None

    def chat(self, user_id: str, message: str) -> str: 

        prompt = self._build_prompt(user_id, message)

        llm_response = openai.chat.completions.create(
            model=self.model,   
            messages=[{"role": "user", "content": prompt}]
        )

        assistant_msg = llm_response.choices[0].message.content

         
        tool_call = self._detect_tool_call(assistant_msg)
        if tool_call:
            tool_name = tool_call["tool"]
            params = tool_call.get("params", {})

            tool_result = self.tools.run(tool_name, params) 

            final_response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": json.dumps(tool_call)},
                    {"role": "user", "content": f"Tool result: {json.dumps(tool_result)}"}
                ]
            )

            assistant_msg = final_response.choices[0].message.content
 
        self.session_memory.setdefault(user_id, []).append({
            "user": message,
            "assistant": assistant_msg
        })

        return assistant_msg
