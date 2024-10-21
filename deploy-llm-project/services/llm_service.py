"""
Handle logic for processing LLM requests.
"""

import time

from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, HuggingFacePipeline
from torch import cuda as torch_cuda

from utils.constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
from utils.cuda_utils import calculate_layer_count

embeddings_model_name = "all-MiniLM-L6-v2"  # os.environ.get("EMBEDDINGS_MODEL_NAME")

model_n_batch = 8
target_source_chunks = 4
model_n_ctx = 2000


def process_llm_request(request):
    # Your LLM processing logic here
    pass


def call_model(query, model_type, hide_source):
    # Parse the command line arguments
    # args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )

    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    # callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM/mnt/nas1/nba055-2/privateGPTpp/models/llama-2-7b-chat.ggmlv3.q4_0.bin
    match model_type:
        case "LlamaCpp":
            # llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            llm = LlamaCpp(
                model_path=r"/data/privateGPTpp/models/llama-2-7b-chat.ggmlv3.q4_0.bin",
                n_ctx=model_n_ctx,
                verbose=False,
                n_gpu_layers=calculate_layer_count(),
            )
        case "GPT4All":
            # llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            llm = GPT4All(
                model="data/privateGPTpp/models/ggml-gpt4all-j-v1.3-groovy.bin",
                backend="gptj",
                verbose=False,
            )
        case "Minitron":
            llm = HuggingFacePipeline.from_model_id(
                model_id="/data/assgn_2/privateGPTpp/models/Minitron-4B-Base",
                task="text-generation",
                device=1,
                model_kwargs={
                    "trust_remote_code": True,
                    "torch_dtype": "auto",
                    "max_length": model_n_ctx,
                },
            )
        case "phi":
            # llm = HuggingFacePipeline.from_model_id(model_id='/data/privateGPTpp/models/phi-1_5',task="text-generation",
            #                            model_kwargs={"trust_remote_code": True, "torch_dtype": "auto", "max_length":model_n_ctx})
            llm = HuggingFacePipeline.from_model_id(
                model_id="/data/assgn_2/privateGPTpp/models/dolly-v2-3b",
                task="text-generation",
                model_kwargs={
                    "trust_remote_code": True,
                    "torch_dtype": "auto",
                    "max_length": model_n_ctx,
                },
            )
        case "codegeex2":
            llm = HuggingFacePipeline.from_model_id(
                model_id="/data/privateGPTpp/models/codegeex2-6b",
                task="text-generation",
                device=1,
                model_kwargs={
                    "trust_remote_code": True,
                    "torch_dtype": "auto",
                    "max_length": model_n_ctx,
                },
            )
        case "codellama":
            llm = HuggingFacePipeline.from_model_id(
                model_id="/data/privateGPTpp/models/CodeLlama-7b-hf",
                task="text-generation",
                device=1,
                model_kwargs={
                    "trust_remote_code": True,
                    "torch_dtype": "auto",
                    "max_length": model_n_ctx,
                },
            )
        case "vicuna":
            llm = HuggingFacePipeline.from_model_id(
                model_id="/data/privateGPTpp/models/vicuna-7b-v1.5",
                task="text-generation",
                device=1,
                model_kwargs={
                    "trust_remote_code": True,
                    "torch_dtype": "auto",
                    "max_length": model_n_ctx,
                },
            )
        case _default:
            # raise exception if model_type is not supported
            raise Exception(
                f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All"
            )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not hide_source,
    )
    # Interactive questions and answers

    # Get the answer from the chain
    start = time.time()
    res = qa(query)
    answer, docs = res["result"], [] if hide_source else res["source_documents"]
    end = time.time()

    # Print the result
    """print("\n\n> Question:")
    print(query)
    print(f"\n> Answer (took {round(end - start, 2)} s.):")
    print(answer)"""

    # Print the relevant sources used for the answer
    sources = []
    for document in docs:
        # print("\n> " + document.metadata["source"] + ":")
        # print(document.page_content)
        # Append source and page content to sources list
        sources.append(document.metadata["source"] + ":" + document.page_content)

    return answer, sources
