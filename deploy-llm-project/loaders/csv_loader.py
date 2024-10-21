from langchain_community.document_loaders import CSVLoader


def load_csv(file_path):
    return CSVLoader(file_path)
