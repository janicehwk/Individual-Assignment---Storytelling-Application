# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Pipeline:
#   1. Image Captioning — Salesforce/blip-image-captioning-base[cite: 1]
#   2. Story Generation — ajibawa-2023/Young-Children-Storyteller-Mistral-7B
#   3. Text-to-Speech   — facebook/mms-tts-eng
# ============================================================================

# ── Import Part ─────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import scipy.io.wavfile
import torch

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(url):
    """
    Generate a text caption from an uploaded image.
    Uses the BLIP image captioning model.[cite: 1]
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(url).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    output = cap_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def text2story(scenario):
    """
    Generate a narrative (50-100 words) for kids aged 3-10.[cite: 1]
    Uses Mistral-7B fine-tuned for children's storytelling.
    """
    # System prompt to guide the model toward age-appropriate content[cite: 1]
    prompt = f"Write a magical and happy short story for a young child about: {scenario}. Once upon a time,"

    # Initialize the storytelling pipeline
    story_pipe = pipeline(
        "text-generation", 
        model="ajibawa-2023/Young-Children-Storyteller-Mistral-7B",
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    raw_story = ""

    # Generation loop to aim for the 50-100 words requirement[cite: 1]
    for attempt in range(3):
        story_results = story_pipe(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1
        )

        raw_story = story_results[0]["generated_text"]
        
        # Ensure the story starts at the narrative beginning
        if "Once upon a time," in raw_story:
            raw_story = "Once upon a time," + raw_story.split("Once upon a time,")[1]

        word_count = len(raw_story.split())
        if 50 <= word_count <= 100:
            break

    # Final logic to ensure strict adherence to the 100-word maximum[cite: 1]
    words = raw_story.split()
    if len(words) > 100:
        words = words[:100]
        raw_story = " ".join(words) + "..."

    return raw_story


def text2audio(story_text):
    """
    Convert a story string into an audio format for an engaging experience.[cite: 1]
    Uses Facebook's MMS-TTS model.
    """
    tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")

    # MMS model works best with lowercase input
    clean_text = story_text.lower()
    speech = tts_pipe(clean_text)

    # Save to a temporary WAV file for Streamlit playback[cite: 1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(
        temp_file.name,
        rate=speech["sampling_rate"],
        data=speech["audio"][0]
    )

    return temp_file.name


# ── Main Part ───────────────────────────────────────────────────────────────
def main():
    """
    Main function for the Streamlit UI and app orchestration.[cite: 1]
    """

    # ── Page Configuration ──────────────────────────────────────────────
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    # ── Custom CSS for Kid-Friendly UI[cite: 1] ─────────────────────────
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .step-label { font-size: 1.2rem; font-weight: 700; color: #2D3436; background: #FFEAA7; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 10px; }
        .story-box { background: #FFFFFF; border: 4px dashed #6C5CE7; border-radius: 20px; padding: 25px; font-size: 1.15rem; line-height: 1.8; color: #2D3436; margin: 15px 0; }
        .caption-box { background: #DFE6E9; border-radius: 15px; padding: 15px 20px; font-size: 1.05rem; color: #2D3436; margin: 10px 0; }
        .fun-footer { text-align: center; color: #636E72; font-size: 0.9rem; margin-top: 40px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'story_finished' not in st.session_state:
        st.session_state.story_finished = False

    st.title("🪄 Magic Story Machine 🪄")
    st.subheader("Upload a picture and watch it turn into a story! 📖✨")

    # ── Step 1: Image Upload[cite: 1] ──────────────────────────────────
    st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a fun image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

        # ── Step 2: Captioning[cite: 1] ────────────────────────────────
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture..."):
            scenario = img2text(file_path)
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # ── Step 3: Story Generation[cite: 1] ──────────────────────────
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        with st.spinner("✍️ Writing a magical story..."):
            story = text2story(scenario)
        st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)

        # ── Step 4: Text-to-Speech[cite: 1] ────────────────────────────
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready..."):
            audio_file_path = text2audio(story)

        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

        st.balloons()
        st.success("🎉 Your story is ready! 🎧")
        st.session_state.story_finished = True

        if st.session_state.story_finished:
            if st.button("🔄 Create Another Story!"):
                st.session_state.uploader_key += 1
                st.session_state.story_finished = False
                st.rerun()

    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
