import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

# Load model and tokenizer once
model = AutoModel.from_pretrained("laion/clap-htsat-unfused", dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")

def get_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate semantic similarity between two texts using CLAP model.
    
    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare
        
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
    # Tokenize both texts
    inputs1 = tokenizer([text1], padding=True, return_tensors="pt")
    inputs2 = tokenizer([text2], padding=True, return_tensors="pt")

    with torch.no_grad():
        # Get embeddings for both texts
        text_features1 = model.get_text_features(**inputs1)
        text_features2 = model.get_text_features(**inputs2)
        
        # Normalize the embeddings
        text_features1 = F.normalize(text_features1, p=2, dim=1)
        text_features2 = F.normalize(text_features2, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.mm(text_features1, text_features2.transpose(0, 1))

    return similarity.item()

if __name__ == "__main__":
    # Test pairs
    test_pairs = [
        ("sad boy", "Very slow, gravelly, exhausted senior male audio. A deep, cynical voice, narrating his last case with profound weariness and dry wit. Very slow, monotone pace."),
        ("sad girl", "Very slow, gravelly, exhausted senior male audio. A deep, cynical voice, narrating his last case with profound weariness and dry wit. Very slow, monotone pace."),
        ("happy boy", "Very slow, gravelly, exhausted senior male audio. A deep, cynical voice, narrating his last case with profound weariness and dry wit. Very slow, monotone pace."),
        ("happy girl", "Very slow, gravelly, exhausted senior male audio. A deep, cynical voice, narrating his last case with profound weariness and dry wit. Very slow, monotone pace."),
        # ("happy singing voice", "joyful vocal performance"),
        # ("old man speaking", "elderly male narrator"),
        # ("sad female voice", "melancholy woman talking"),
        # ("excited announcer", "calm whisper"),
        # ("middle aged male singing", "young boy humming"),
        # ("the sound of rain", "the sound of thunder"),
        # ("classical music", "rock music")
    ]
    
    print("CLAP Text Similarity Test Results:")
    print("=" * 50)
    
    for text1, text2 in test_pairs:
        similarity = get_text_similarity(text1, text2)
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Similarity score: {similarity:.4f}")
        print("-" * 30)