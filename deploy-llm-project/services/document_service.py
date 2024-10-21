import os
import glob

from typing import List
from multiprocessing import Pool
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from utils.constants import LOADER_MAPPING, PERSIST_DIRECTORY, CHROMA_SETTINGS

# embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 500
chunk_overlap = 50


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, docs in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                results.extend(docs)
                pbar.update()

    return results


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {PERSIST_DIRECTORY}")
    # Print the name of the document that is currently being processed
    for file_name in os.listdir(PERSIST_DIRECTORY):
        if file_name not in ignored_files:
            print(f"Processing {file_name}")
            break
    documents = load_documents(PERSIST_DIRECTORY, ignored_files)
    print(documents)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {PERSIST_DIRECTORY}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Check if the vector store exists in the specified directory
    """
    index_path = os.path.join(persist_directory, "index")
    if os.path.exists(index_path):
        if os.path.isdir(index_path):
            if os.listdir(index_path):
                return True
    return False


def ingest():
    embeddings_model_name = (
        "all-MiniLM-L6-v2"  # os.environ.get("EMBEDDINGS_MODEL_NAME")
    )
    persist_directory = "data/privateGPTpp/db"  # os.environ.get('PERSIST_DIRECTORY')

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings,
            client_settings=CHROMA_SETTINGS,
        )
        collection = db.get()
        texts = process_documents(
            [metadata["source"] for metadata in collection["metadatas"]]
        )
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(texts)
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(
            texts,
            embeddings,
            persist_directory=persist_directory,
            client_settings=CHROMA_SETTINGS,
        )
    db.persist()
    db = None
    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
