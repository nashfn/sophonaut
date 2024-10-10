import chainlit as cl
import openai
import os
import base64

api_key = os.getenv("OPENAI_API_KEY")


endpoint_url = "https://api.openai.com/v1"
#endpoint_url = f"https://api.runpod.ai/v2/{runpod_serverless_id}/openai/v1"

client = openai.AsyncClient(api_key=api_key, base_url=endpoint_url)

# https://platform.openai.com/docs/models/gpt-4o

model_kwargs = {
    "model": "gpt-4o",
    "temperature": 1.2,
    "max_tokens": 500
}


@cl.on_message
async def on_message(message: cl.Message):
    # Record the AI's response in the history
    message_history = cl.user_session.get("message_history", [])

    # Image processing
    images = [file for file in message.elements if "image" in file.mime] if message.elements else []

    if images:
        # Read one image after another
        with open(images[0].path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode('utf-8')
            message_history.append({"role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content if message.content else "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }]})
    else:
        message_history.append({"role": "user", "content": message.content})

    cl.user_session.set("message_history", message_history)

    response_message = cl.Message(content="")
    await response_message.send()

    # full message history
    stream = await client.chat.completions.create(messages=message_history, stream=True, **model_kwargs)

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
    print (f"{message_history}")

    # https://platform.openai.com/docs/guides/chat-completions/response-format
    #response_content = response.choices[0].message.content

    # send a response back to the user
    #await cl.Message(content=response_content).send()

