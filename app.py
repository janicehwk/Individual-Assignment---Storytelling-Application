# ============================================================================
# Program Title: Magic Story Machine - A Storytelling App for Kids
# Description:   A kid-friendly Streamlit application that turns uploaded
#                images into fun audio stories for children aged 3-10.
# Pipeline:
#   1. Image Captioning — Salesforce/blip-image-captioning-base
#   2. Story Generation — facebook/opt-350m
#   3. Text-to-Speech   — facebook/mms-tts-eng (Hugging Face TTS)
# ============================================================================

# ── Import Part ─────────────────────────────────────────────────────────────
import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import scipy.io.wavfile


# ── Function Part ───────────────────────────────────────────────────────────

def img2text(url):
    """
    Generate a text caption from an uploaded image.
    Uses the BLIP image captioning model loaded directly via
    BlipProcessor and BlipForConditionalGeneration for compatibility.

    Parameters:
        url (str): File path of the uploaded image.
    Returns:
        str: A short caption describing what is in the image.
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
    Generate a kid-friendly story (50-100 words) from an image caption.
    Uses Meta's OPT-350M text generation model. Includes a retry loop to
    strictly enforce the 50-word minimum requirement.

    Parameters:
        scenario (str): A short image caption to build the story around.
    Returns:
        str: A fun story between 50-100 words for children aged 3-10.
    """
    # Build a longer prompt to give the model more context to continue from
    prompt = (
        f"Once upon a time, {scenario}. "
        f"It was a beautiful sunny day. "
        f"Everyone was happy and excited. "
        f"The adventure was about to begin. "
    )

    story_pipe = pipeline(
        "text-generation",
        model="facebook/opt-350m"
    )

    raw_story = ""

    # Try up to 5 times to guarantee at least 50 words
    for attempt in range(5):
        story_results = story_pipe(
            prompt,
            max_new_tokens=200,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.85 + (attempt * 0.1),
            repetition_penalty=1.2
        )

        raw_story = story_results[0]["generated_text"]

        word_count = len(raw_story.split())
        if word_count >= 50:
            break

    # If still under 50 words after retries, pad with a closing sentence
    if len(raw_story.split()) < 50:
        raw_story += (
            " They laughed and played together all day long. "
            "When the sun began to set, they knew it was time to go home. "
            "But they promised to meet again tomorrow for another adventure. "
            "And they all lived happily ever after. The end."
        )

    # Trim to stay within 100 words maximum
    words = raw_story.split()
    if len(words) > 100:
        words = words[:100]
        trimmed_story = " ".join(words)

        last_punctuation = max(
            trimmed_story.rfind('.'),
            trimmed_story.rfind('!'),
            trimmed_story.rfind('?')
        )

        if last_punctuation != -1:
            trimmed_story = trimmed_story[:last_punctuation + 1]
        else:
            trimmed_story += "..."

        return trimmed_story

    return raw_story


def text2audio(story_text):
    """
    Convert a story string into a natural audio file using
    Facebook's MMS-TTS model from Hugging Face.

    Parameters:
        story_text (str): The story text to convert to speech.
    Returns:
        str: File path of the generated audio (.wav) file.
    """
    tts_pipe = pipeline("text-to-speech", model="facebook/mms-tts-eng")

    # The MMS model requires text to be lowercase to read it properly
    clean_text = story_text.lower()

    # Generate the audio array
    speech = tts_pipe(clean_text)

    # Save to a temporary WAV file so Streamlit can play it
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
    Main function that runs the entire Streamlit application.
    Handles page configuration, kid-friendly UI styling, session state,
    image upload, and orchestrates the three-stage pipeline:
    image → caption → story → audio playback.
    """

    # ── Page Configuration ──────────────────────────────────────────────
    st.set_page_config(
        page_title="Magic Story Machine",
        page_icon="🪄",
        layout="centered"
    )

    # ── Custom CSS for Kid-Friendly UI ──────────────────────────────────
    st.markdown("""
    <style>
        .stApp { background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%); }
        .step-label { font-size: 1.2rem; font-weight: 700; color: #2D3436; background: #FFEAA7; padding: 8px 16px; border-radius: 20px; display: inline-block; margin-bottom: 10px; }
        .story-box { background: #FFFFFF; border: 4px dashed #6C5CE7; border-radius: 20px; padding: 25px; font-size: 1.15rem; line-height: 1.8; color: #2D3436; margin: 15px 0; }
        .caption-box { background: #DFE6E9; border-radius: 15px; padding: 15px 20px; font-size: 1.05rem; color: #2D3436; margin: 10px 0; }
        .fun-footer { text-align: center; color: #636E72; font-size: 0.9rem; margin-top: 40px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

    # ── Initialize Session States for Reset Functionality ───────────────
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    if 'story_finished' not in st.session_state:
        st.session_state.story_finished = False

    # ── App Header ──────────────────────────────────────────────────────
    st.title("🪄 Magic Story Machine 🪄")
    st.subheader("Upload a picture and watch it turn into a story! 📖✨")

    # ── Step 1: Image Upload ────────────────────────────────────────────
    st.markdown('<p class="step-label">📸 Step 1: Pick a Picture!</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a fun image...",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        # Save uploaded file locally for the model to read
        bytes_data = uploaded_file.getvalue()
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(bytes_data)

        st.image(uploaded_file, caption="🖼️ Your awesome picture!", use_container_width=True)

        # ── Step 2: Image → Caption ─────────────────────────────────────
        st.markdown('<p class="step-label">🔍 Step 2: What\'s in your picture?</p>', unsafe_allow_html=True)
        with st.spinner("🧐 Looking at your picture really carefully..."):
            scenario = img2text(file_path)
        st.markdown(f'<div class="caption-box">I see: <strong>{scenario}</strong></div>', unsafe_allow_html=True)

        # ── Step 3: Caption → Story ─────────────────────────────────────
        st.markdown('<p class="step-label">📝 Step 3: Story time!</p>', unsafe_allow_html=True)
        st.text('Generating a story...')

        with st.spinner("✍️ Writing a magical story just for you..."):
            story = text2story(scenario)

        st.write(f"**Story:** {story}")

        # ── Step 4: Story → Audio ───────────────────────────────────────
        st.markdown('<p class="step-label">🔊 Step 4: Listen to your story!</p>', unsafe_allow_html=True)
        with st.spinner("🎵 Getting the story ready to read aloud..."):
            audio_file_path = text2audio(story)

        # Play the WAV file generated by the Facebook MMS model
        with open(audio_file_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

        st.balloons()
        st.success("🎉 Your story is ready! Press play to listen! 🎧")
        st.session_state.story_finished = True

        # ── Create Another Story Flow ───────────────────────────────────
        if st.session_state.story_finished:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Create Another Story!"):
                widget_key = f"uploader_{st.session_state.uploader_key}"
                if widget_key in st.session_state:
                    del st.session_state[widget_key]

                st.session_state.uploader_key += 1
                st.session_state.story_finished = False
                st.rerun()

    st.markdown('<p class="fun-footer">Made with ❤️ for little storytellers everywhere 🌈</p>', unsafe_allow_html=True)


# ── Run the App ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
