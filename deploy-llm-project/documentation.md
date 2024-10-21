# Explanation of the app

To run the app, simply do
`python3 app.py`

## Loaders
This directory contains modules responsible for loading different types of data.

- csv_loader.py: This file contains functions and classes to load data from CSV files.
- email_loader.py: This file contains functions and classes to load data from email sources.

## Routes
This directory is intended to contain the route (blueprint) definitions for the application.
Its like a higher level abstraction of the app, where the actual app will make POST-request to start the predictions etc.

- main_routes.py: Contains all the blueprints and routes

## Services
This directory is intended to contain service modules. Services usually encapsulate business logic and interact with data models or external APIs.

- document_service.py: Primarily provides functions to load documents from a specified directory. 
- llm_service.py: Responsible for handling requests to a language model (LLM) and managing the interaction with different LLMs.

## Templates
This directory contains the actual HTMl to render the app on the web

## Utils
This directory contains different kinds of helper functions 

- constants.py: Names of directories, model configurations. This you can modify to accomodate where the data is etc.
- cuda_utils.py: Mainly to calculate cpu layers etc. to use the app with GPU

