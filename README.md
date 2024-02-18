This is an example of a Retrival Augmented Generation model for German language PDFs

As a model I used DRXD1000/Phoenix LLM which is a trained mistral model and possibly the best German speaking model out there. I used LlamaCpp to run the model locally on my Macbook M1 for which I needed a GGUF version of the model.

Just downlod a model from huggingface as a GGUF file, install llama.cpp and all python dependencies (mostly langchain), and run rag.py
