from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings


from llama_parse import LlamaParse
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# Llamaparse: https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb

VECTOR_INDEX_PATH = "./index/vector_index_001.json"
PDF_PATH = "./data/CLC-Workbook.pdf"

from copy import deepcopy
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex

from llama_index.core.node_parser import MarkdownElementNodeParser


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


def run_main():
    index = create_index_from_pdf(PDF_PATH)
    index.save_to_disk(VECTOR_INDEX_PATH)

if __name__ == "__main__":
    run_main()