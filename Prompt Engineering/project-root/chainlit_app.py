import chainlit as cl
from src.orchestrator import Orchestrator

orch = Orchestrator()

@cl.on_message
async def main(message: cl.Message):
    user_id = str(message.author) or "anon"
    user_msg = message.content
    result = orch.run(user_id, user_msg)
    text = result.get("response", "متأسفم، پاسخی موجود نیست.")
    if result.get("tools_used"):
        text += f"\n\nTools used: {result.get('tools_used')}"
    await cl.Message(content=text).send()
