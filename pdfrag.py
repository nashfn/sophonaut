from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings


from llama_parse import LlamaParse

documents = LlamaParse(result_type="markdown").load_data("./apple_2021_10k.pdf")