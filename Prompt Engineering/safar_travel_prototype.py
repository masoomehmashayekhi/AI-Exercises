# Project: Safar Travel - Prototype
# File layout included below as separate file sections. Save each section as the named path.

# -----------------------------
# FILE: requirements.txt
# -----------------------------
# chainlit is used for the chat frontend prototype
# fastapi is optional; not used in this minimal prototype
# language detection and date parsing
chainlit>=1.10.0
langdetect>=1.0.9
python-dateutil>=2.8.2
pydantic>=1.10.7

# -----------------------------
# FILE: .env.example
# -----------------------------
# Example environment variables
GROQ_API_KEY=
TAVILY_API_KEY=
CHAINLIT_TOKEN=

# -----------------------------
# FILE: run.py
# -----------------------------
"""Run script for local development.
Usage: python run.py
This will start the Chainlit app (see chainlit_app.py)
"""
import os
import subprocess
import sys

if __name__ == '__main__':
    # Launch chainlit with the entry file chainlit_app.py
    cmd = [
        sys.executable, '-m', 'chainlit', 'run', 'chainlit_app.py',
        '--dev'
    ]
    print('Running:', ' '.join(cmd))
    subprocess.run(cmd)

# -----------------------------
# FILE: chainlit_app.py
# -----------------------------
"""Chainlit frontend that connects to the manager orchestrator.
This is a lightweight example that demonstrates conversation flow and function-calling simulation.
"""
import chainlit as cl
from src.manager import SafarManager

manager = SafarManager()

@cl.on_chat_start
async def start():
    await cl.Message(content="سلام! من دستیار سفَر هستم — چطور کمکتون کنم؟\n(برای English، کافیست به انگلیسی بنویسید)").send()

@cl.on_message
async def main(message: cl.Message):
    # Pass user message to manager, get responses as messages
    responses = await manager.handle_user_message(message.content)
    for r in responses:
        if isinstance(r, dict) and r.get('type') == 'card':
            await cl.Message(content=r.get('text')).send()
        else:
            await cl.Message(content=r).send()

# -----------------------------
# FILE: prompts.py
# -----------------------------
"""Prompts and prompt templates used by the orchestrator.
The system prompt is deliberately explicit about tool usage, language handling, and date processing.
"""
SYSTEM_PROMPT = {
    'role': 'system',
    'content': (
        "You are Safar Travel's professional multilingual customer service agent. "
        "Always detect the user's language and reply in the same language. "
        "You MUST NOT invent booking data: always call the ticket tool to read/create/cancel bookings. "
        "When live-searching destinations, call the tavily_search tool. "
        "For company policy or FAQ, use the rag_lookup tool. "
        "If a date phrase is relative (e.g., 'tomorrow', 'next Sunday', 'پس فردا'), normalize it to ISO-8601 for tool calls. "
        "Tone: polite, concise, helpful. For Persian, be culturally respectful and use proper polite phrases."
    )
}

FEW_SHOT_EXAMPLES = [
    # Short few-shot examples that the manager can show to the model (if used)
]

# -----------------------------
# FILE: src/__init__.py
# -----------------------------
# Package init

# -----------------------------
# FILE: src/tools.py
# -----------------------------
"""Tool implementations: Tavily search mock, Chroma RAG mock, and Ticket Management simulated API.
These are synchronous/simple for the prototype; in production they'd be asynchronous and robust.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid

# In-memory ticket store
_TICKETS = {}

# Simple city normalization (expandable)
_CITY_ALIASES = {
    'تهران': 'Tehran',
    'شیراز': 'Shiraz',
    'اصفهان': 'Isfahan',
    'مشهد': 'Mashhad'
}


def normalize_city(name: str) -> str:
    name = name.strip()
    return _CITY_ALIASES.get(name, name)


# ----------------
# Ticket API (simulated)
# ----------------

def book_ticket(origin: str, destination: str, departure_date: str, passengers: int, passenger_info: List[Dict[str, Any]], travel_class: str = 'economy') -> Dict[str, Any]:
    """Create a simulated booking and return booking details."""
    booking_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat() + 'Z'
    price_per = 1200000  # placeholder IRR price
    total_price = price_per * passengers

    ticket = {
        'booking_id': booking_id,
        'origin': normalize_city(origin),
        'destination': normalize_city(destination),
        'departure_date': departure_date,
        'passengers': passengers,
        'passenger_info': passenger_info,
        'class': travel_class,
        'price': total_price,
        'created_at': now,
        'status': 'CONFIRMED'
    }
    _TICKETS[booking_id] = ticket
    return {
        'success': True,
        'data': ticket
    }


def get_booking(booking_id: str) -> Dict[str, Any]:
    t = _TICKETS.get(booking_id)
    if not t:
        return {'success': False, 'error': f'Booking {booking_id} not found'}
    return {'success': True, 'data': t}


def cancel_booking(booking_id: str, cancellation_date_iso: Optional[str] = None) -> Dict[str, Any]:
    t = _TICKETS.get(booking_id)
    if not t:
        return {'success': False, 'error': 'not_found'}

    # Simple refund policy: if canceled >=48 hours before departure => full refund, else 50%
    dep = datetime.fromisoformat(t['departure_date'])
    now = datetime.fromisoformat(cancellation_date_iso) if cancellation_date_iso else datetime.utcnow()
    delta = dep - now
    if delta >= timedelta(hours=48):
        refund = t['price']
        penalty = 0
    else:
        refund = int(t['price'] * 0.5)
        penalty = t['price'] - refund

    t['status'] = 'CANCELLED'
    t['cancelled_at'] = now.isoformat()
    t['refund'] = refund
    t['penalty'] = penalty
    return {'success': True, 'data': {'booking_id': booking_id, 'refund': refund, 'penalty': penalty}}


# ----------------
# Tavily Search (mock)
# ----------------

def tavily_search(query: str, location_filter: Optional[str] = None, max_results: int = 3) -> List[Dict[str, Any]]:
    """Mock implementation returning structured search results for Iranian cities."""
    # Very simplistic mock data; in production this calls Tavily API
    samples = [
        {'title': 'Persepolis - Shiraz', 'snippet': 'Historical ruins near Shiraz, best visited in spring.', 'url': 'https://example.com/persepolis', 'publish_date': '2024-03-10'},
        {'title': 'Nasir al-Mulk Mosque', 'snippet': 'Famous colorful mosque in Shiraz (Nasir al-Mulk).', 'url': 'https://example.com/nasir', 'publish_date': '2023-11-02'},
        {'title': 'Eram Garden - Shiraz', 'snippet': 'Famous garden with seasonal blooms.', 'url': 'https://example.com/eram', 'publish_date': '2022-07-15'},
    ]
    return samples[:max_results]


# ----------------
# Chroma RAG mock (very small)
# ----------------

def rag_lookup(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    # For prototype, return some canned policy snippets
    docs = [
        {'id': 'policy-1', 'title': 'Cancellation Policy', 'content': 'Full refund if cancelled more than 48 hours before departure.'},
        {'id': 'policy-2', 'title': 'Booking Policy', 'content': 'Ticket issuance happens after payment confirmation.'}
    ]
    return docs[:top_k]

# -----------------------------
# FILE: src/manager.py
# -----------------------------
"""Core orchestrator: receives user messages, decides which tool(s) to call, performs minimal NLU, and returns user-facing messages.
This prototype uses heuristic NLU to detect intent and extract simple slots. In the course, you can replace heuristics with LLM-driven extraction.
"""
from typing import List, Dict, Any
from langdetect import detect
from dateutil import parser as dateparser
from datetime import datetime
import re

from .tools import book_ticket, get_booking, cancel_booking, tavily_search, rag_lookup
from prompts import SYSTEM_PROMPT


class SafarManager:
    def __init__(self):
        # session memory per conversation (in-memory)
        self.sessions: Dict[str, Dict[str, Any]] = {}

    async def handle_user_message(self, text: str) -> List[Any]:
        # 1. language detection
        try:
            lang = detect(text)
        except Exception:
            lang = 'fa' if re.search(r'[\u0600-\u06FF]', text) else 'en'

        # 2. intent detection (very simple heuristics for prototype)
        intent, slots = self.simple_intent_and_slots(text)

        # 3. route based on intent
        if intent == 'book_ticket':
            # ensure required slots
            missing = [s for s in ['origin', 'destination', 'departure_date', 'passengers'] if s not in slots]
            if missing:
                return [self._translate("لطفاً مبدأ، مقصد، تاریخ حرکت و تعداد مسافر را بفرمایید." if lang.startswith('fa') else "Please provide origin, destination, travel date and number of passengers.", lang)]

            # normalize date to ISO
            try:
                dep = self.normalize_date(slots['departure_date'])
            except Exception:
                return [self._translate("تاریخ نامعتبر است. لطفاً به فرمت YYYY-MM-DD بفرستید یا عباراتی مثل "نزدیک فردا" را واضح‌تر بنویسید.", lang)]

            passenger_info = slots.get('passenger_info', [])
            resp = book_ticket(slots['origin'], slots['destination'], dep, int(slots['passengers']), passenger_info, travel_class=slots.get('class', 'economy'))
            if resp['success']:
                data = resp['data']
                msg = self._translate(f"رزرو با موفقیت انجام شد. کد رزرو: {data['booking_id']}، قیمت کل: {data['price']} تومان.\nبرای مشاهده جزئیات بیشتر 'نمایش بلیط {data['booking_id']}' را بفرستید." if lang.startswith('fa') else f"Booking successful. Booking ID: {data['booking_id']}, total price: {data['price']}.\nTo view details send 'show booking {data['booking_id']}'.", lang)
                return [msg]
            else:
                return [self._translate('خطا در انجام رزرو.' if lang.startswith('fa') else 'Failed to create booking.', lang)]

        elif intent == 'get_booking':
            if 'booking_id' not in slots:
                return [self._translate('لطفاً کد رزرو را ارسال کنید.' if lang.startswith('fa') else 'Please provide booking ID.', lang)]
            resp = get_booking(slots['booking_id'])
            if resp['success']:
                d = resp['data']
                msg = self._translate(f"جزئیات رزرو {d['booking_id']}: {d['origin']} → {d['destination']} در تاریخ {d['departure_date']}. وضعیت: {d['status']}" if lang.startswith('fa') else f"Booking {d['booking_id']}: {d['origin']} -> {d['destination']} on {d['departure_date']}. Status: {d['status']}", lang)
                return [msg]
            else:
                return [self._translate('رزروی پیدا نشد.' if lang.startswith('fa') else 'Booking not found.', lang)]

        elif intent == 'cancel_booking':
            if 'booking_id' not in slots:
                return [self._translate('کد رزرو را بفرستید تا لغو را انجام دهم.' if lang.startswith('fa') else 'Please provide booking ID to cancel.', lang)]
            # cancellation date = now
            resp = cancel_booking(slots['booking_id'], cancellation_date_iso=datetime.utcnow().isoformat())
            if resp['success']:
                d = resp['data']
                msg = self._translate(f"رزرو {d['booking_id']} کنسل شد. مبلغ مرجوعی: {d['refund']} تومان. کارمزد: {d['penalty']}" if lang.startswith('fa') else f"Booking {d['booking_id']} cancelled. Refund: {d['refund']}. Penalty: {d['penalty']}", lang)
                return [msg]
            else:
                return [self._translate('خطا: رزرو پیدا نشد.' if lang.startswith('fa') else 'Error: booking not found.', lang)]

        elif intent == 'search_destination':
            query = text
            results = tavily_search(query, max_results=3)
            messages = []
            for r in results:
                messages.append(self._translate(f"{r['title']} - {r['snippet']} (منبع: {r['url']})" if lang.startswith('fa') else f"{r['title']} - {r['snippet']} (source: {r['url']})", lang))
            return messages

        elif intent == 'rag_query':
            docs = rag_lookup(text, top_k=2)
            msgs = [self._translate(f"{d['title']}: {d['content']}" if lang.startswith('fa') else f"{d['title']}: {d['content']}", lang) for d in docs]
            return msgs

        else:
            # fallback: try to be helpful and offer options
            return [self._translate('متوجه نشدم؛ می‌تونم در رزرو، لغو یا جستجوی مقصد کمکتون کنم. لطفاً بگید دنبال چی هستید.' if lang.startswith('fa') else "I didn't understand. I can help with booking, cancelling, getting booking info, or searching destinations. What would you like?", lang)]

    def simple_intent_and_slots(self, text: str) -> (str, Dict[str, Any]):
        t = text.lower()
        slots: Dict[str, Any] = {}
        # booking intent
        if any(w in t for w in ['بلیط', 'رزرو', 'book', 'reserve']):
            # origin/destination naive extraction (look for 'از X به Y' or 'from X to Y')
            m = re.search(r'از\s+([\w\u0600-\u06FF\s]+)\s+به\s+([\w\u0600-\u06FF\s]+)', text)
            if m:
                slots['origin'] = m.group(1).strip()
                slots['destination'] = m.group(2).strip()
            m2 = re.search(r'([0-9]{4}-[0-9]{2}-[0-9]{2})', text)
            if m2:
                slots['departure_date'] = m2.group(1)
            # passengers
            m3 = re.search(r'(\d+)\s*(نفر|passenger|people)', text)
            if m3:
                slots['passengers'] = m3.group(1)
            else:
                slots['passengers'] = 1
            return 'book_ticket', slots

        if any(w in t for w in ['نمایش بلیط', 'show booking', 'booking', 'کد رزرو']):
            m = re.search(r'([A-Za-z0-9-]{4,})', text)
            if m:
                slots['booking_id'] = m.group(1)
            return 'get_booking', slots

        if any(w in t for w in ['کنسل', 'لغو', 'cancel']):
            m = re.search(r'([A-Za-z0-9-]{4,})', text)
            if m:
                slots['booking_id'] = m.group(1)
            return 'cancel_booking', slots

        if any(w in t for w in ['پیشنهاد', 'کجا برم', 'where to', 'suggest']):
            return 'search_destination', {}

        if any(w in t for w in ['سیاست', 'policy', 'قانون', 'faq']):
            return 'rag_query', {}

        return 'unknown', {}

    def normalize_date(self, text: str) -> str:
        # Try ISO first
        try:
            dt = dateparser.parse(text)
            return dt.date().isoformat()
        except Exception:
            # fallback: if Persian relative phrases used, handle basic cases
            text = text.strip()
            if text in ['فردا', 'tomorrow']:
                return (datetime.utcnow().date() + timedelta(days=1)).isoformat()
            if text in ['پس فردا', 'پس‌فردا']:
                return (datetime.utcnow().date() + timedelta(days=2)).isoformat()
            # else raise
            raise ValueError('invalid date')

    def _translate(self, text: str, lang: str) -> str:
        # In prototype, messages are already created in both languages; so return original
        return text

# -----------------------------
# FILE: data/README.txt
# -----------------------------
# Place your RAG text files (policies, FAQs) in this folder. The prototype rag_lookup uses canned data.

# -----------------------------
# End of project prototype file
# -----------------------------
