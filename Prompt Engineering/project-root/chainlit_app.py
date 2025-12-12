import chainlit as cl
from src.orchestrator import Orchestrator

orch = Orchestrator()

@cl.on_chat_start
async def start():
    await cl.Message(
        content="سلام!  من دستیار سفر شما هستم.\nچطور می‌تونم کمکتون کنم؟"
    ).send()


@cl.on_message
async def main(message: cl.Message):

    user_id = "anon" 
    user_msg = message.content

    result = orch.run(user_id, user_msg)

    response_text = result.get("response", "متاسفم، پاسخی موجود نیست.")
    tools_used = result.get("tools_used", [])
    sources = result.get("sources", [])

    display_text = response_text
    if tools_used:
        display_text += f"\n\n Tools used: {tools_used}"
    if sources:
        display_text += f"\n Sources: {sources}"

    await cl.Message(content=display_text).send()