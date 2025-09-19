from sentence_transformers import SentenceTransformer, util

# Load a pre-trained model.
# This model is lightweight and effective for semantic similarity tasks.
# It's best to load this once and reuse it if you're calling the function multiple times.
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an internet connection and the sentence-transformers library is installed.")
    model = None

def get_similarity(text1: str, text2: str) -> float:
    """
    Calculates the semantic similarity between two texts using a sentence-transformer model.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.

    Returns:
        float: A similarity score between 0.0 and 1.0, where 1.0 means identical.
               Returns 0.0 if the model is not loaded.
    """
    if not model:
        print("Model is not loaded. Returning similarity of 0.0")
        return 0.0

    # 1. Encode the texts into numerical vectors (embeddings)
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)

    # 2. Compute the cosine similarity between the two embeddings
    # Cosine similarity measures the cosine of the angle between two vectors,
    # resulting in a value between -1 and 1. For these models, it's typically 0 to 1.
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2)

    # The result is a tensor, so we extract the single float value
    return cosine_score.item()

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: High similarity
    prompt1 = "An old man with a deep, raspy voice."
    prompt2 = "An elderly gentleman speaking hoarsely."
    similarity1 = get_similarity(prompt1, prompt2)
    print(f"Prompt 1: '{prompt1}'")
    print(f"Prompt 2: '{prompt2}'")
    print(f"Similarity Score: {similarity1:.4f}\n") # Expected: ~0.85+

    # Example 2: Moderate similarity
    prompt3 = "A happy child laughing."
    prompt4 = "A joyful person."
    similarity2 = get_similarity(prompt3, prompt4)
    print(f"Prompt 3: '{prompt3}'")
    print(f"Prompt 4: '{prompt4}'")
    print(f"Similarity Score: {similarity2:.4f}\n") # Expected: ~0.60-0.70

    # Example 3: Low similarity
    prompt5 = "A sports announcer shouting excitedly."
    prompt6 = "A quiet, whispering narrator."
    similarity3 = get_similarity(prompt5, prompt6)
    print(f"Prompt 5: '{prompt5}'")
    print(f"Prompt 6: '{prompt6}'")
    print(f"Similarity Score: {similarity3:.4f}\n") # Expected: <0.30
