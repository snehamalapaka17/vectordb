from transformers import pipeline

# Load a small model for paraphrasing/normalization
try:
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", use_fast=False)
except Exception as e:
    print(f"Error loading paraphraser model: {e}")
    print("Trying alternative model...")
    try:
        paraphraser = pipeline("text2text-generation", model="t5-small", use_fast=False)
    except Exception as e2:
        print(f"Error loading alternative model: {e2}")
        paraphraser = None

def normalize_text(text: str) -> str:
    if paraphraser is None:
        print("Paraphraser model not available, returning original text")
        return text
    
    prompt = f"paraphrase: {text}"
    try:
        out = paraphraser(prompt, max_length=32, num_return_sequences=1, do_sample=False)
        return out[0]['generated_text']
    except Exception as e:
        print(f"Error during paraphrasing: {e}")
        return text

# Example
if paraphraser is not None:
    print(normalize_text("old talking tree"))
    # -> "ancient tree that speaks"
else:
    print("Paraphraser not available")