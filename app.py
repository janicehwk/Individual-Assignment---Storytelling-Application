# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Pipeline:
#   1. Image Captioning — Salesforce/blip-image-captioning-base
#   2. Story Generation — roneneldan/TinyStories-33M
#   3. Text-to-Speech   — facebook/mms-tts-eng (Hugging Face TTS)
# ============================================================================

import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import scipy.io.wavfile

# ── Function Part ───────────────────────────────────────────────────────────

def img2text(url):
    """Generate a text caption from an uploaded image.[cite: 1]"""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    cap_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(url).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    output = cap_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def text2story(scenario):
    """Generate a kid-friendly story (50-100 words) from a caption.[cite: 1]"""
    technical_words = ["illustration", "vector", "drawing", "image", "picture", "graphic"]
    clean_scenario = scenario
    for word in technical_words:
        clean_scenario = clean_scenario.replace(word, "").strip()

    prompt = (
        f"Once upon a time, there were {clean_scenario}. "
        f"It was a beautiful sunny day. "
        f"Everyone was happy and excited. "
        f"The adventure was about to begin. "
    )

    story_pipe = pipeline("text-generation", model="roneneldan/TinyStories-33M")
    raw_story = ""

    for attempt in range(5):
        story_results = story_pipe(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.85 + (attempt * 0.1),
            repetition_penalty=1.2
        )
        raw_story = story_results[0]["generated_text"]
        if len(raw_story.split()) >= 50:
            break

    if len(raw_story.split()) < 50:
        raw_story += " They all lived happily ever after. The end."

    words = raw_story.split()
    if len(words) > 100:
        words = words[:100]
        trimmed_story = " ".join(words)
        last_punct = max(trimmed_story.rfind('.'), trimmed_story.rfind('!'), trimmed_story.rfind('?'))
        return trimmed_story[:last_punct + 1] if last_punct != -1 else trimmed_story + "..."
    return raw_story

def text2audio(story_text):
    """Convert story text into audio using Hugging Face TTS.[cite: 1]"""
    tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    speech = tts_pipe(story_text.lower())
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(temp_file.name, rate=speech["sampling_rate"], data=speech["audio"][0])
    return temp_file.name

# ── Main Part ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Magic Story Machine", page_icon="🪄", layout="centered")

    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .step-label { font-size: 1.2rem; font-weight: 700; color: #2D3436; background: #FFEAA7; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 10px; }
        .story-box { background: #FFFFFF; border: 4px dashed #6C5CE7; border-radius: 20px; padding: 25px; font-size: 1.15rem; line-height: 1.8; color: #2D3436; margin: 15px 0; }
        .caption-box { background: #DFE6E9; border-radius: 15px; padding: 15px 20px; font-size: 1.05rem; color: #2D3436; margin: 10px 0; }
        .fun-footer { text-align: center; color: #636E72; font-size: 0.9rem; margin-top: 40px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize uploader key
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    st.title("🪄 Magic Story Machine 🪄")
    st.subheader("Upload a picture and watch it turn into a story! 📖✨")

    st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a fun image...", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # The pipeline only runs if a file is present and we aren't in a reset state
    if uploaded_file is not None:
        # Step 2: Caption[cite: 1]
        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture..."):
            scenario = img2text(uploaded_file)
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # Step 3: Story[cite: 1]
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        with st.spinner("✍️ Writing a magical story..."):
            story = text2story(scenario)
        st.write(f"**Story:** {story}")

        # Step 4: Audio[cite: 1]
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready..."):
            audio_path = text2audio(story)
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        st.balloons()
        st.success("🎉 Your story is ready!")

        # Reset button at the bottom of the page[cite: 1]
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Create Another Story!"):
            # Wipe all session data to force a fresh landing page state[cite: 1]
            st.session_state.uploader_key += 1
            st.rerun()

    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()        f"Once upon a time, there were {clean_scenario}. "
        f"It was a beautiful sunny day. "
        f"Everyone was happy and excited. "
        f"The adventure was about to begin. "
    )

    story_pipe = pipeline("text-generation", model="roneneldan/TinyStories-33M")
    raw_story = ""

    for attempt in range(5):
        story_results = story_pipe(
            prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.85 + (attempt * 0.1),
            repetition_penalty=1.2
        )
        raw_story = story_results[0]["generated_text"]
        if len(raw_story.split()) >= 50:
            break

    if len(raw_story.split()) < 50:
        raw_story += " They all lived happily ever after. The end."

    words = raw_story.split()
    if len(words) > 100:
        words = words[:100]
        trimmed_story = " ".join(words)
        last_punct = max(trimmed_story.rfind('.'), trimmed_story.rfind('!'), trimmed_story.rfind('?'))
        return trimmed_story[:last_punct + 1] if last_punct != -1 else trimmed_story + "..."
    return raw_story

def text2audio(story_text):
    """Convert story text into audio using Hugging Face TTS.[cite: 1]"""
    tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")
    speech = tts_pipe(story_text.lower())
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    scipy.io.wavfile.write(temp_file.name, rate=speech["sampling_rate"], data=speech["audio"][0])
    return temp_file.name

# ── Main Part ───────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Magic Story Machine", page_icon="🪄", layout="centered")

    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .step-label { font-size: 1.2rem; font-weight: 700; color: #2D3436; background: #FFEAA7; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 10px; }
        .story-box { background: #FFFFFF; border: 4px dashed #6C5CE7; border-radius: 20px; padding: 25px; font-size: 1.15rem; line-height: 1.8; color: #2D3436; margin: 15px 0; }
        .caption-box { background: #DFE6E9; border-radius: 15px; padding: 15px 20px; font-size: 1.05rem; color: #2D3436; margin: 10px 0; }
        .fun-footer { text-align: center; color: #636E72; font-size: 0.9rem; margin-top: 40px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize uploader key
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    st.title("🪄 Magic Story Machine 🪄")
    st.subheader("Upload a picture and watch it turn into a story! 📖✨")

    st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a fun image...", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    # The pipeline only runs if a file is present and we aren't in a reset state
    if uploaded_file is not None:
        # Step 2: Caption[cite: 1]
        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture..."):
            scenario = img2text(uploaded_file)
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # Step 3: Story[cite: 1]
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        with st.spinner("✍️ Writing a magical story..."):
            story = text2story(scenario)
        st.write(f"**Story:** {story}")

        # Step 4: Audio[cite: 1]
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready..."):
            audio_path = text2audio(story)
        with open(audio_path, "rb") as f:
            st.audio(f.read(), format="audio/wav")

        st.balloons()
        st.success("🎉 Your story is ready!")

        # Reset button at the bottom of the page[cite: 1]
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Create Another Story!"):
            # Wipe all session data to force a fresh landing page state[cite: 1]
            st.session_state.clear()
            st.rerun()

    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
