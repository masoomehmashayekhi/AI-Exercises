import chainlit as cl
from src.manager import SafarManager

manager = SafarManager()

@cl.on_chat_start
async def start():
    await cl.Message(content="سلام! من دستیار سفَر هستم — چطور کمکتون کنم؟\n(Hi, I'm your travel assistant. How can I help you?)").send()

@cl.on_message
async def main(message: cl.Message): 
    responses = await manager.handle_user_message(message.content)
    for r in responses:
        if isinstance(r, dict) and r.get('type') == 'card':
            await cl.Message(content=r.get('text')).send()
        else:
            await cl.Message(content=r).send()
