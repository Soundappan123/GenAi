import streamlit as st
from langchain.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn as nn
AIzaSyDggmjNRml6aay0xB5Xhxscm6A-snmN3AM
# Define Mytryoshka Loss
class MytryoshkaLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MytryoshkaLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)
    
    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        positive_distance = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
        negative_distance = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)
        return self.loss_fn(positive_distance, negative_distance, torch.ones_like(positive_distance))

# Define Recall@k
def recall_at_k(retrieved, relevant, k=10):
    retrieved_at_k = set(retrieved[:k])
    relevant_set = set(relevant)
    return len(retrieved_at_k.intersection(relevant_set)) / len(relevant_set)

# Define NDCG@k
def ndcg_at_k(retrieved, relevant, k=10):
    dcg = 0
    idcg = sum([1 / np.log2(i + 2) for i in range(len(relevant))])
    for i, doc in enumerate(retrieved[:k]):
        if doc in relevant:
            dcg += 1 / np.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0

# Process PDF
def process_pdf(pdf_file):
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    return documents

# Embed text
def embed_text(text, embedding_model):
    return np.array(embedding_model.encode(text))

# Semantic Chunking
def semantic_chunking(documents, embedding_model, threshold=0.7):
    chunks = []
    for doc in documents:
        text = doc.page_content
        sentences = text.split('. ')  # Sentence-level granularity
        current_chunk = []
        previous_embedding = None

        for sentence in sentences:
            if not sentence.strip():
                continue
            current_embedding = embed_text(sentence, embedding_model)
            if previous_embedding is None:
                current_chunk.append(sentence)
                previous_embedding = current_embedding
                continue

            similarity = cosine_similarity(
                previous_embedding.reshape(1, -1),
                current_embedding.reshape(1, -1)
            )[0][0]

            if similarity >= threshold:
                current_chunk.append(sentence)
            else:
                chunks.append('. '.join(current_chunk))
                current_chunk = [sentence]

            previous_embedding = current_embedding

        if current_chunk:
            chunks.append('. '.join(current_chunk))

    return chunks

# Evaluate Semantic Chunking
def evaluate_chunking(chunks, query, embedding_model, ground_truth, top_k=5):
    query_embedding = embedding_model.encode([query])
    chunk_embeddings = [embedding_model.encode(chunk) for chunk in chunks]

    # Calculate similarities
    similarities = cosine_similarity(query_embedding, np.array(chunk_embeddings))[0]
    retrieved = [chunks[i] for i in similarities.argsort()[-top_k:][::-1]]

    # Metrics
    recall = recall_at_k(retrieved, ground_truth, k=top_k)
    ndcg = ndcg_at_k(retrieved, ground_truth, k=top_k)
    mrr = 1 if any([chunk in ground_truth for chunk in retrieved[:1]]) else 0

    return retrieved, recall, ndcg, mrr

# Streamlit UI
st.title("PDF Semantic Chunking and Evaluation")

# Upload PDF file
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    # Initialize SentenceTransformers model
    embedding_model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")

    # Process and display the PDF
    documents = process_pdf(pdf_file)

    if documents:
        st.write("Successfully loaded PDF. Starting semantic chunking...")
        chunks = semantic_chunking(documents, embedding_model, threshold=0.7)

        if chunks:
            st.write("Semantic Chunking Complete!")
            for i, chunk in enumerate(chunks):
                st.write(f"### Chunk {i + 1}:\n{chunk}\n")

            # Query input
            query = st.text_input("Enter Query:", "L298 Motor Driver:")
            ground_truth = st.text_area("Enter Ground Truth (comma separated)", 
                                        "Learn Buddy is a path-following robot combining robotics and AI for lab assistance.").split(',')

            # Evaluate
            if query:
                retrieved_chunks, recall, ndcg, mrr = evaluate_chunking(chunks, query, embedding_model, ground_truth, top_k=5)
                
                # Display metrics
                st.write("### Evaluation Metrics")
                st.write(f"MRR: {mrr:.4f}")
                st.write(f"Recall@5: {recall:.4f}")
                st.write(f"NDCG@5: {ndcg:.4f}")

                # Display retrieved chunks
                st.write("### Retrieved Chunks")
                for i, chunk in enumerate(retrieved_chunks):
                    st.write(f"Chunk {i + 1}: {chunk}")
        else:
            st.write("No chunks generated.")
