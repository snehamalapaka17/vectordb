import pandas as pd

# Read the CSV file
df = pd.read_csv('/home/sneha/code/vectordb/Public-library-19sep.csv')

# Extract voice titles and descriptions as separate lists
voice_titles = df['voice title'].tolist()
voice_audios = df['reference_audio'].tolist()
voice_preview_audios = df['preview_audio (tts_mgm)'].tolist()
voice_descriptions = df['voice description'].tolist()

def get_voice_from_description(combined_text: str) -> dict:
    """
    Given a *combined* "Title. Description" string, finds and returns 
    the corresponding voice title, reference audio, and preview audio.
    
    Args:
        combined_text (str): The "Title. Description" string to search for
        
    Returns:
        dict: A dictionary containing 'title', 'reference_audio', 'preview_audio', 
              and 'description', or None if not found
    """
    try:
        # We must re-create the combined string for all entries to find a match
        for i, (title, desc) in enumerate(zip(voice_titles, voice_descriptions)):
            expected_combined_text = f"{title}. {desc}"
            if expected_combined_text == combined_text:
                # Found it! Return the data from that index.
                return {
                    'title': voice_titles[i],
                    'reference_audio': voice_audios[i],
                    'preview_audio': voice_preview_audios[i],
                    'description': voice_descriptions[i] # Also return the clean description
                }
        
        # If no match is found
        return None
        
    except ValueError:
        # This block is for safety, but the loop above is better
        return None

if __name__ == "__main__":

    print(df.head())

    print(f"Number of voice titles: {len(voice_titles)}")
    print(f"Number of voice descriptions: {len(voice_descriptions)}")

    # Display first few examples
    print("\nFirst 5 voice titles:")
    for i, title in enumerate(voice_titles[:5]):
        print(f"{i+1}. {title}")

    print("\nFirst 5 voice descriptions:")
    for i, description in enumerate(voice_descriptions[:5]):
        print(f"{i+1}. {description}")
        
    # Test the NEW function
    print("\nTesting get_voice_from_description function:")
    if voice_descriptions:
        # Create a test combined string
        test_combined = f"{voice_titles[0]}. {voice_descriptions[0]}"
        print(f"Testing with: {test_combined}")
        
        result_data = get_voice_from_description(test_combined)
        
        if result_data:
            print("Test successful. Found data:")
            print(f"  Title: {result_data['title']}")
            print(f"  Description: {result_data['description']}")
            print(f"  Audio: {result_data['reference_audio']}")
        else:
            print("Test FAILED. Function did not find a match.")