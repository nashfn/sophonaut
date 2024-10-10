import chainlit as cl
import openai
import os
import base64
import json
from dotenv import load_dotenv

from pdfrag import load_vector_index, get_retriever, retrieve_docs_for_query
from pdfrag import VECTOR_INDEX_PATH, LLM_QUERY_PROMPT


load_dotenv()
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


BASIC_PROMPT = """
You are a helpful assistant that can answer questions based on a Tony Robbins, "Creating Lasting Change": Seven steps to mastering leadership. 
Be available as a helpful assistant to answer the user's questions. Use the given tools to fetch answers to the user's questions.
"""

BASE_TOOLS =  [
        {
            "type": "function",
            "function": {
                "name": "queryCoach",
                "description": "User's query relating to the Tony Robbins material on 'Creating Lasting Change'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's quqery as a sentence.",
                        },
                    },
                    "required": ["query",],
                    "additionalProperties": False,
                },
            }
        },
]


pdf_vector_index = load_vector_index(VECTOR_INDEX_PATH)
retriever_engine = get_retriever(pdf_vector_index)

async def complete_query(response_message, query):
    retrieved_context = retrieve_docs_for_query(retriever_engine, query)
    print(f"Retrieved context = {retrieved_context}")
    temp_message_history = [{"role": "system", "content": LLM_QUERY_PROMPT.format(
        context_str=retrieved_context, query_str=query,)},]
    stream = await client.chat.completions.create(messages=temp_message_history, stream=True, **model_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)  


@cl.on_chat_start
def on_chat_start():    
    message_history = [{"role": "system", "content": BASIC_PROMPT}]
    cl.user_session.set("message_history", message_history)

async def execute_function(response_message, function_name, arguments):
    if function_name:
            print(f"DEBUG: function_name: {function_name}")
            print(" Arguments: ", arguments)
    if function_name == "queryCoach":
        arguments_dict = json.loads(arguments)
        query = arguments_dict.get("query")
        if query:
            print("Received user query as:", query)
            await complete_query(response_message, query)


async def process_user_query(stream, response_message):
        function_list = []
        function_name = ""
        arguments = ""
        async for part in stream:
            if part.choices[0].delta.tool_calls:
                tool_call = part.choices[0].delta.tool_calls[0]
                function_name_delta = tool_call.function.name or ""
                arguments_delta = tool_call.function.arguments or ""
                #print(f"function_name_delta = {function_name_delta}")
                
                if function_name_delta and arguments:
                    #print(f"Adding {function_name_delta} and {arguments} to the stack.")
                    function_list.append((function_name, arguments)) # prev function.
                    arguments = ""
                    function_name = ""
                function_name += function_name_delta
                arguments += arguments_delta
                #print(f"arguments delta = {arguments_delta}")
        
            if token := part.choices[0].delta.content or "":
                await response_message.stream_token(token)        

        if function_name and arguments:
            function_list.append((function_name, arguments)) # prev function.
            
        for function_name, arguments in function_list:
            await execute_function(response_message, function_name, arguments)


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
    stream = await client.chat.completions.create(messages=message_history, stream=True, tools=BASE_TOOLS, **model_kwargs)
    await process_user_query(stream, response_message)


    # async for part in stream:
    #     if token := part.choices[0].delta.content or "":
    #         await response_message.stream_token(token)

    await response_message.update()

    # Record the AI's response in the history
    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)
    print (f"{message_history}")

    # https://platform.openai.com/docs/guides/chat-completions/response-format
    #response_content = response.choices[0].message.content

    # send a response back to the user
    #await cl.Message(content=response_content).send()

