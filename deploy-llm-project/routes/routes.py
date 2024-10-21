"""
Handle main routes for the application.
"""

import os

from flask import Blueprint, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from services.llm_service import process_llm_request, call_model
from services.document_service import ingest

source_directory = r"../source_documents/cleanup"

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/process", methods=["POST"])
def process():
    # Your processing logic here
    return process_llm_request(request)


@main_bp.route("/upload", methods=["POST"])
def upload():
    # Get the filename
    filename = request.files["file"].filename

    # Upload the file to the source directory

    file = request.files["file"]  # .read().decode("latin-1")
    # print(file)
    # Save the file to the source directory
    """os.chdir(source_directory)
    with open(filename, "w") as f:
        f.write(file)"""
    file.save("/data/privateGPTpp/source_documents/" + (file.filename))
    ingest()

    # Return a message to the json file
    # return {'message': 'File uploaded successfully'}
    # return a message to be displayed on the "/" webpage and not the "/upload" webpage
    return redirect(url_for("hello"))


@main_bp.route("/predict", methods=["POST"])
def predict():
    # text = str(request.form['text'])
    # Get the text from the json file
    data = request.get_json()
    text = data["prompt"]
    # text = data['input']
    # Select model from drop down list of index.html
    model_type = data["model"]
    print(text)
    print(model_type)

    # Check if the source directory is empty
    if not os.listdir(source_directory):
        print("Source directory is empty. Please upload a file first.")

    answer, sources = call_model(text, model_type, hide_source=False)
    print(sources)
    # From each of the elements in the sources list, split the string at the first colon
    sources = [source.split(":", 1) for source in sources]
    # Concatenate the sources list to a string
    sources = "\n\n".join([source[1] for source in sources])
    # Concatenate the sources string to the answer string and add Source: to the beginning of the sources string
    answer = answer + "\n\nSources :\n\n" + sources
    # Return the answer and sources as a dict which can be read in json format in javascript
    return {"answer": answer}
