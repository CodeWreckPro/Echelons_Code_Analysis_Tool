from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from app.models.evolution import ChangeMetrics

class EmbeddingService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

    def get_embedding(self, text: str):
        """Generate an embedding for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def analyze_commit_importance(
        self, 
        commit_message: str, 
        changes: List[ChangeMetrics]
    ) -> float:
        """Analyze the semantic importance of a commit."""
        # Combine commit message and change information
        change_summary = " ".join([f"{c.change_type} in {c.file_path}" for c in changes])
        text_to_embed = f"{commit_message} {change_summary}"
        
        # Generate embedding
        embedding = self.get_embedding(text_to_embed)
        
        # Calculate importance score (e.g., based on embedding norm)
        importance = torch.norm(torch.tensor(embedding)).item()
        
        return importance

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate the semantic similarity between two texts."""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            torch.tensor(embedding1).unsqueeze(0),
            torch.tensor(embedding2).unsqueeze(0)
        ).item()
        
        return cos_sim