import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import scipy.io.wavfile
import torch

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(url):
    """Generate a text caption from an uploaded image."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(url).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = cap_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def text2story(scenario):
    """Generate a kid-friendly story (50-100 words) using TinyStories."""
    # TinyStories is much better for the 1GB RAM limit on Streamlit Cloud
    story_pipe = pipeline("text-generation", model="roneneldan/TinyStories-33M")
    
    prompt = f"Once upon a time, {scenario}. It was a magical day. "
    
    story_results = story_pipe(
        prompt, 
        max_new_tokens=150, 
        do_sample=True, 
        temperature=0.7
    )
    
    full_text = story_results[0]["generated_text"]
    
    # Ensure word count logic for assignment requirements[cite: 1]
    words = full_text.split()
    if len(words) > 100:
        words = words[:100]
        full_text = " ".join(words) + "."
    
    return full_text

def text2audio(story_text):
    """Convert story to speech using a lightweight TTS model[cite: 1]."""
    tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    speech = tts_pipe(story_text.lower())
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(temp_file.name, rate=speech["sampling_rate"], data=speech["audio"][0])
    return temp_file.name

# ── Main Part ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Magic Story Machine", page_icon="🪄")
    
    st.title("🪄 Magic Story Machine 🪄")
    st.subheader("Turn your pictures into stories for kids![cite: 1]")

    uploaded_file = st.file_uploader("Pick a Picture!", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        with open("temp_img.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, use_container_width=True)
        
        with st.spinner("Analyzing image..."):
            caption = img2text("temp_img.jpg")
            st.write(f"**I see:** {caption}")
            
        with st.spinner("Writing story..."):
            story = text2story(caption)
            st.info(story)
            
        with st.spinner("Preparing audio..."):
            audio_path = text2audio(story)
            st.audio(audio_path)
            st.balloons()

if __name__ == "__main__":
    main()
