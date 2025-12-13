import chainlit as cl
from src.orchestrator import Orchestrator

orch = Orchestrator()

@cl.on_message
async def main(message: cl.Message): 
    try:
        user_id = str(message.author) or "anon"
        user_msg = message.content
        result = orch.run(user_id, user_msg)
        text = result.get("response", "متأسفم، پاسخی موجود نیست.")
        if result.get("tools_used"):
            text += f"\n\nTools used: {result.get('tools_used')}"
        await cl.Message(content=text, author="Assistant").send()
    except Exception as e:
        error_message = f"Unfortunately, an error occurred: {str(e)}"
        if "api_key" in str(e).lower():
            error_message = "Please provide a valid Groq API key in the .env file."
        elif "quota" in str(e).lower():
            error_message = "API usage quota has been reached. Please try again later."
        await cl.Message(
            content=error_message,
            author="System"
        ).send()
    
