import chromadb
# We need this import from database.py for the gradio_search function
from database import get_voice_from_description
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Any
import uuid
import os
import gradio as gr
from tqdm import tqdm

# Load all models
clap_model = None
clap_tokenizer = None
minilm_model = None
mpnet_model = None

# Load CLAP model
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused", dtype=torch.float16)
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
    print("Loaded CLAP model for audio-text similarity")
except Exception as e:
    print(f"Error loading CLAP model: {e}")
    print("Please ensure you have torch and transformers libraries installed.")

# Load MiniLM model
try:
    minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Loaded all-MiniLM-L6-v2 model for text similarity")
except Exception as e:
    print(f"Error loading MiniLM model: {e}")
    print("Please ensure you have an internet connection and the sentence-transformers library is installed.")

# Load MPNet model
try:
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    print("Loaded all-mpnet-base-v2 model for text similarity")
except Exception as e:
    print(f"Error loading MPNet model: {e}")
    print("Please ensure you have an internet connection and the sentence-transformers library is installed.")

# Initialize ChromaDB client with persistent storage
db_path = "./chroma_db"
os.makedirs(db_path, exist_ok=True)
client = chromadb.PersistentClient(path=db_path)

def get_or_create_collection(collection_name: str = "text_embeddings"):
    """
    Get or create a ChromaDB collection.
    
    Args:
        collection_name (str): Name of the collection
        
    Returns:
        Collection: ChromaDB collection object
    """
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    return collection

def get_collection_name(model_type: str) -> str:
    """
    Get collection name based on model type.
    
    Args:
        model_type (str): Either 'clap', 'minilm', or 'mpnet'
        
    Returns:
        str: Collection name
    """
    if model_type.lower() == 'clap':
        return "clap_embeddings"
    elif model_type.lower() == 'mpnet':
        return "mpnet_embeddings"
    else:
        return "minilm_embeddings"

def add_texts_to_db(texts: List[str], model_type: str = "minilm", ids: Optional[List[str]] = None):
    """
    Add texts to the ChromaDB database using specified model.
    
    Args:
        texts (List[str]): List of text strings to add
        model_type (str): Either 'clap', 'minilm', or 'mpnet'
        ids (Optional[List[str]]): List of IDs for each text. If None, UUIDs will be generated
    """
    collection_name = get_collection_name(model_type)
    collection = get_or_create_collection(collection_name)
    
    # Generate IDs if not provided
    if ids is None:
        ids = [str(uuid.uuid4()) for _ in texts]
    
    # Generate embeddings based on model type
    if model_type.lower() == 'clap':
        if not clap_model or not clap_tokenizer:
            print("CLAP model is not loaded. Cannot add texts to database.")
            return
        
        import torch
        embeddings = []
        print(f"Generating CLAP embeddings for {len(texts)} texts...")
        for text in tqdm(texts, desc="Processing texts with CLAP"):
            inputs = clap_tokenizer([text], padding=True, return_tensors="pt")
            with torch.no_grad():
                text_features = clap_model.get_text_features(**inputs)
                embeddings.append(text_features.squeeze().tolist())
    elif model_type.lower() == 'mpnet':
        if not mpnet_model:
            print("MPNet model is not loaded. Cannot add texts to database.")
            return
        
        # Use sentence transformer
        print(f"Generating MPNet embeddings for {len(texts)} texts...")
        embeddings = mpnet_model.encode(texts, show_progress_bar=True).tolist()
    else:
        if not minilm_model:
            print("MiniLM model is not loaded. Cannot add texts to database.")
            return
        
        # Use sentence transformer
        print(f"Generating MiniLM embeddings for {len(texts)} texts...")
        embeddings = minilm_model.encode(texts, show_progress_bar=True).tolist()
    
    # Add to collection
    print(f"Adding {len(texts)} texts to collection '{collection_name}'...")
    collection.add(
        embeddings=embeddings,
        documents=texts,
        ids=ids
    )
    
    print(f"Added {len(texts)} texts to collection '{collection_name}' using {model_type.upper()} model")

def search_similar_texts(query: str, model_type: str = "minilm", n_results: int = 5) -> Dict[str, Any]:
    """
    Search for similar texts in the database using specified model.
    
    Args:
        query (str): Query text to search for
        model_type (str): Either 'clap', 'minilm', or 'mpnet'
        n_results (int): Number of results to return
        
    Returns:
        Dict[str, Any]: Search results containing documents and distances
    """
    collection_name = get_collection_name(model_type)
    collection = get_or_create_collection(collection_name)
    
    # Generate query embedding based on model type
    if model_type.lower() == 'clap':
        if not clap_model or not clap_tokenizer:
            print("CLAP model is not loaded. Cannot perform search.")
            return {"documents": [], "distances": []}
        
        import torch
        inputs = clap_tokenizer([query], padding=True, return_tensors="pt")
        with torch.no_grad():
            query_embedding = clap_model.get_text_features(**inputs)
            query_embedding = query_embedding.squeeze().tolist()
    elif model_type.lower() == 'mpnet':
        if not mpnet_model:
            print("MPNet model is not loaded. Cannot perform search.")
            return {"documents": [], "distances": []}
        
        # Use sentence transformer
        query_embedding = mpnet_model.encode([query]).tolist()[0]
    else:
        if not minilm_model:
            print("MiniLM model is not loaded. Cannot perform search.")
            return {"documents": [], "distances": []}
        
        # Use sentence transformer
        query_embedding = minilm_model.encode([query]).tolist()[0]
    
    # Perform search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

def get_similarity(text1: str, text2: str, model_type: str = "minilm") -> float:
    """
    Calculates the semantic similarity between two texts using the specified model.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        model_type (str): Either 'clap', 'minilm', or 'mpnet'

    Returns:
        float: A similarity score between 0.0 and 1.0, where 1.0 means identical.
               Returns 0.0 if the model is not loaded.
    """
    if model_type.lower() == 'clap':
        if not clap_model or not clap_tokenizer:
            print("CLAP model is not loaded. Returning similarity of 0.0")
            return 0.0
        
        # Use CLAP model
        import torch
        import torch.nn.functional as F
        
        # Tokenize both texts
        inputs1 = clap_tokenizer([text1], padding=True, return_tensors="pt")
        inputs2 = clap_tokenizer([text2], padding=True, return_tensors="pt")

        with torch.no_grad():
            # Get embeddings for both texts
            text_features1 = clap_model.get_text_features(**inputs1)
            text_features2 = clap_model.get_text_features(**inputs2)
            
            # Normalize the embeddings
            text_features1 = F.normalize(text_features1, p=2, dim=1)
            text_features2 = F.normalize(text_features2, p=2, dim=1)
            
            # Calculate cosine similarity
            similarity = torch.mm(text_features1, text_features2.transpose(0, 1))

        return similarity.item()
    elif model_type.lower() == 'mpnet':
        if not mpnet_model:
            print("MPNet model is not loaded. Returning similarity of 0.0")
            return 0.0
        
        # Use sentence transformer model
        embedding1 = mpnet_model.encode(text1, convert_to_tensor=True)
        embedding2 = mpnet_model.encode(text2, convert_to_tensor=True)
        from sentence_transformers import util
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_score.item()
    else:
        if not minilm_model:
            print("MiniLM model is not loaded. Returning similarity of 0.0")
            return 0.0
        
        # Use sentence transformer model
        embedding1 = minilm_model.encode(text1, convert_to_tensor=True)
        embedding2 = minilm_model.encode(text2, convert_to_tensor=True)
        from sentence_transformers import util
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
        return cosine_score.item()

# --- MODIFIED FUNCTION ---
def gradio_search(query: str, model_type: str) -> str:
    """
    Gradio interface function to search for similar texts.
    
    Args:
        query (str): Query text to search for
        model_type (str): Either 'CLAP', 'MiniLM', or 'MPNet'
        
    Returns:
        str: Formatted search results
    """
    if not query.strip():
        return "Please enter a search query."
    
    # Convert display name to internal format
    model_type_internal = model_type.lower().replace('minilm', 'minilm').replace('mpnet', 'mpnet')
    
    results = search_similar_texts(query, model_type=model_type_internal, n_results=5)
    
    if not results['documents'] or not results['documents'][0]:
        return f"No results found using {model_type} model. Make sure the database is populated for this model."
    
    output = f"Top 5 matches for: '{query}' using {model_type} model\n\n"
    
    for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        similarity_score = 1 - distance  # Convert distance to similarity
        
        # Get the corresponding voice title
        try:
            # This function now comes from database.py and is
            # designed to find data from the combined 'doc' string
            voice_info = get_voice_from_description(doc)
            
            if voice_info:
                # --- THIS IS THE UPDATED FORMAT ---
                output += f"{i+1}. Title: {voice_info['title']}\n"
                output += f"   Audio: {voice_info['reference_audio']}\n"
                output += f"   Preview Audio: {voice_info['preview_audio']}\n"
                # This line is changed from 'doc' to 'voice_info['description']'
                output += f"   Description: {voice_info['description']}\n"
            else:
                # Fallback if the lookup fails for some reason
                output += f"{i+1}. {doc}\n"
        except ImportError:
            output += f"{i+1}. {doc}\n"
        
        output += f"   Similarity: {similarity_score:.4f} (Distance: {distance:.4f})\n\n"
    
    return output

# --- MODIFIED FUNCTION ---
def initialize_database():
    """Initialize the database with combined titles and descriptions."""
    try:
        # Import both titles and descriptions
        from database import voice_titles, voice_descriptions
        
        print("Combining titles and descriptions for embedding...")
        if len(voice_titles) != len(voice_descriptions):
            print(f"Warning: Mismatch in lengths. Titles: {len(voice_titles)}, Descriptions: {len(voice_descriptions)}")
        
        # Create the combined text for embedding
        combined_texts = [
            f"{title}. {desc}" 
            for title, desc in zip(voice_titles, voice_descriptions)
        ]
        
        if not combined_texts:
            print("No data found in database.py to initialize.")
            return
            
        print(f"Created {len(combined_texts)} combined text entries.")
        if combined_texts:
            print(f"Example: '{combined_texts[0]}'")

        # Initialize MiniLM collection
        minilm_collection = get_or_create_collection(get_collection_name("minilm"))
        minilm_count = minilm_collection.count()
        if minilm_count == 0:
            print("Initializing MiniLM database with combined texts...")
            # Pass the new 'combined_texts' list
            add_texts_to_db(combined_texts, model_type="minilm")
            print(f"MiniLM database initialized with {len(combined_texts)} entries.")
        else:
            print(f"MiniLM database already contains {minilm_count} entries.")
        
        # Initialize MPNet collection if model is available
        if mpnet_model:
            mpnet_collection = get_or_create_collection(get_collection_name("mpnet"))
            mpnet_count = mpnet_collection.count()
            if mpnet_count == 0:
                print("Initializing MPNet database with combined texts...")
                # Pass the new 'combined_texts' list
                add_texts_to_db(combined_texts, model_type="mpnet")
                print(f"MPNet database initialized with {len(combined_texts)} entries.")
            else:
                print(f"MPNet database already contains {mpnet_count} entries.")
        else:
            print("MPNet model not available, skipping MPNet database initialization.")
        
        # Initialize CLAP collection if model is available
        if clap_model and clap_tokenizer:
            clap_collection = get_or_create_collection(get_collection_name("clap"))
            clap_count = clap_collection.count()
            if clap_count == 0:
                print("Initializing CLAP database with combined texts...")
                # Pass the new 'combined_texts' list
                add_texts_to_db(combined_texts, model_type="clap")
                print(f"CLAP database initialized with {len(combined_texts)} entries.")
            else:
                print(f"CLAP database already contains {clap_count} entries.")
        else:
            print("CLAP model not available, skipping CLAP database initialization.")
            
    except ImportError:
        print("Could not import 'voice_titles', 'voice_descriptions', or 'get_voice_from_description'. Database will be empty.")
    except Exception as e:
        print(f"Error initializing database: {e}")

# Create Gradio interface
def create_gradio_app():
    """Create and return the Gradio interface."""
    
    # Initialize database
    initialize_database()
    
    # Determine available models for radio options
    available_models = []
    if minilm_model:
        available_models.append("MiniLM")
    if mpnet_model:
        available_models.append("MPNet")
    if clap_model and clap_tokenizer:
        available_models.append("CLAP")
    
    # Default to MiniLM if available, otherwise first available model
    default_model = "MiniLM" if "MiniLM" in available_models else (available_models[0] if available_models else "MiniLM")
    
    interface = gr.Interface(
        fn=gradio_search,
        inputs=[
            gr.Textbox(
                label="Search Query",
                placeholder="Enter your search query (e.g., 'old person', 'happy voice', 'sad singing')",
                lines=2
            ),
            gr.Radio(
                choices=available_models,
                label="Embedding Model",
                value=default_model,
                info="Choose the embedding model for semantic search. CLAP is specialized for audio-text similarity, MiniLM and MPNet are general-purpose text similarity models."
            )
        ],
        outputs=gr.Textbox(
            label="Top 5 Similar Matches",
            lines=15,
            max_lines=20
        ),
        title="Voice Description Search",
        description="Search for similar voice descriptions using semantic similarity. Choose between CLAP (audio-text specialized), MiniLM (lightweight text), and MPNet (high-quality text) embedding models.",
        examples=[
            ["old person", "MiniLM"],
            ["happy voice", "CLAP"],
            ["sad singing", "CLAP"],
            ["middle aged male", "MPNet"],
            ["female narrator", "MiniLM"],
            ["excited announcer", "CLAP"]
        ]
    )
    
    return interface

# --- Example Usage ---
if __name__ == "__main__":
    # Create and launch Gradio app
    app = create_gradio_app()
    app.launch(share=True, debug=True)