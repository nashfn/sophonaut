from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
import os
from dotenv import load_dotenv
from utils import make_tempdirs


from llama_parse import LlamaParse
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


load_dotenv()

# Llamaparse: https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb

VECTOR_INDEX_PATH = "./index/vector_index_001/"
PDF_PATH = "./data/CLC-Workbook.pdf"

from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

from llama_index.core.node_parser import MarkdownElementNodeParser

LLM_QUERY_PROMPT = """
You are an expert personal coach who can answer questions on specific coaching methods. Answer the given question purely from the text snippets provided below.

Coaching methodology context: {context_str}

User query: {query_str}

"""

def get_page_nodes(docs, separator="\n---\n"):
    """Split each document into page node, by separator."""
    nodes = []
    for doc in docs:
        doc_chunks = doc.text.split(separator)
        for doc_chunk in doc_chunks:
            node = TextNode(
                text=doc_chunk,
                metadata=deepcopy(doc.metadata),
            )
            nodes.append(node)

    return nodes

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
llm = OpenAI(model="gpt-4o")

Settings.llm = llm
Settings.embed_model = embed_model


def create_index_from_pdf(pdf_filepath):
    documents = LlamaParse(result_type="markdown").load_data(pdf_filepath)
    page_nodes = get_page_nodes(documents)
    node_parser = MarkdownElementNodeParser(llm=OpenAI(model="gpt-4-turbo"), num_workers=8)
    nodes = node_parser.get_nodes_from_documents(documents)

    base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
    # dump both indexed tables and page text into the vector index
    recursive_index = VectorStoreIndex(nodes=base_nodes + objects + page_nodes)
    return recursive_index

def save_index_to_file(vector_index, index_json_outfile):
    vector_index.save_to_disk(index_json_outfile)


def load_index_from_file(json_index_filepath):
    index = VectorStoreIndex.load_from_disk('vector_index.json')
    return index


def load_index_from_file(index_json_filepath):
    recursive_index = VectorStoreIndex.load_from_disk(VECTOR_INDEX_PATH)
    return recursive_index

def load_index_create_query_engine(recursive_index):
    reranker = FlagEmbeddingReranker(top_n=5, model="BAAI/bge-reranker-large",)

    recursive_query_engine = recursive_index.as_query_engine(
        similarity_top_k=5, node_postprocessors=[reranker], verbose=True
    )
    return recursive_query_engine


def get_retriever(recursive_index):
    retriever_engine = recursive_index.as_retriever(retrieval_mode='similarity', k=5)
    return retriever_engine


def retrieve_docs_for_query(retriever_engine, query):
    retrieval_results = retriever_engine.retrieve(query)

    retrieved_text = []
    for res_node in retrieval_results:
        retrieved_text.append(res_node.text)

    return retrieved_text 

def query_with_oai(query_str, context):
    text_response = Settings.llm.complete(prompt=LLM_QUERY_PROMPT.format(
        context_str=context, query_str=query_str,),)

    print(text_response.text)
    return text_response.text

def persist_index_to_disk(vector_index, persist_dir):
    make_tempdirs(persist_dir)
    vector_index.storage_context.persist(persist_dir=persist_dir)


def load_vector_index(index_filepath):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_filepath)

    # load index
    index = load_index_from_storage(storage_context)
    return index

def run_main():
    index = create_index_from_pdf(PDF_PATH)
    persist_index_to_disk(index, VECTOR_INDEX_PATH)
    print(f" Index persisted to {VECTOR_INDEX_PATH}.")

if __name__ == "__main__":
    run_main()