from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Load pre-trained model and tokenizer
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", index_name="exact", passages_path="data.txt")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# Define a function for the retrieval-augmented generation
def get_answer(query):
    inputs = tokenizer(query, return_tensors="pt")
    # Get context from retriever
    input_ids = inputs["input_ids"]
    outputs = model.generate(input_ids=input_ids, num_beams=4, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
