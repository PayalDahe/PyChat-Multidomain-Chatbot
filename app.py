import streamlit as st
import torch # Load , Run, TRained, and optimize
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
import io
import soundfile as sf
import speech_recognition as sr

# Import configurations from config.py
from config import (
    DOMAIN_KNOWLEDGE_BASES, GENERAL_CHAT_MODEL_NAME, VQA_MODEL_NAME,
    USE_4BIT_QUANTIZATION, BNB_4BIT_CONFIG, GENERATION_PARAMS
)


# --- 1. Model Loading (Cached for Streamlit) ---
# @st.cache_resource ensures these heavy objects are loaded only once across Streamlit runs.
@st.cache_resource
def get_device_info():
    """Checks for GPU availability and returns device details."""
    if torch.cuda.is_available():
        device_id = 0
        device_name = torch.cuda.get_device_name(device_id)
        device = f"cuda:{device_id}"
        st.sidebar.success(f"CUDA available! Running on: {device_name}")
        return device_id, device_name, device
    else:
        device_id = -1
        device_name = "CPU"
        device = "cpu"
        st.sidebar.warning("CUDA is NOT available. Falling back to CPU.")
        return device_id, device_name, device


@st.cache_resource
def load_general_chat_model(device_id, device_name):
    """Loads the general conversational model (distilgpt2) with optional 4-bit quantization."""
    print(f"Loading general chat model ({GENERAL_CHAT_MODEL_NAME}) on {device_name}...")
    try:
        if device_id != -1 and USE_4BIT_QUANTIZATION:
            model = AutoModelForCausalLM.from_pretrained(
                GENERAL_CHAT_MODEL_NAME,
                quantization_config=BitsAndBytesConfig(**BNB_4BIT_CONFIG),
                device_map="auto"  # Automatically maps model layers to available devices (CPU/GPU)
            )
            tokenizer = AutoTokenizer.from_pretrained(GENERAL_CHAT_MODEL_NAME)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto"
            )
            st.sidebar.success(f"General chat model ({GENERAL_CHAT_MODEL_NAME}) loaded with 4-bit quantization.")
            return pipe
        else:
            pipe = pipeline("text-generation", model=GENERAL_CHAT_MODEL_NAME, device=-1)  # device=-1 for CPU
            st.sidebar.info(f"General chat model ({GENERAL_CHAT_MODEL_NAME}) loaded on {device_name}.")
            return pipe
    except Exception as e:
        st.sidebar.error(f"Error loading {GENERAL_CHAT_MODEL_NAME}: {e}. Falling back to basic CPU pipeline.")
        # Fallback to basic CPU pipeline if advanced settings fail
        return pipeline("text-generation", model=GENERAL_CHAT_MODEL_NAME, device=-1)


@st.cache_resource
def load_vqa_model(device, device_name):
    """Loads a Visual Question Answering (VQA) model."""
    print(f"Loading VQA model ({VQA_MODEL_NAME}) on {device_name}...")
    try:
        # device=0 for cuda:0, device=-1 for CPU
        vqa_pipeline_model = pipeline("visual-question-answering", model=VQA_MODEL_NAME, device=device)
        st.sidebar.success(f"VQA model ({VQA_MODEL_NAME}) loaded.")
        return vqa_pipeline_model
    except Exception as e:
        st.sidebar.error(f"Error loading VQA model ({VQA_MODEL_NAME}): {e}. Multimodal Q&A will be simulated.")
        return None


# Load models once at application start, using cached functions
DEVICE_ID, DEVICE_NAME, DEVICE = get_device_info()
GENERAL_CHAT_PIPELINE = load_general_chat_model(DEVICE_ID, DEVICE_NAME)
VQA_PIPELINE = load_vqa_model(DEVICE, DEVICE_NAME)


# --- 2. Input Preprocessing Functions ---
def process_image_data_for_desc(image_bytes):
    """Processes image bytes to get a simple description (for simulation/fallback)."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        # Simple placeholder description if VQA model is not used or fails
        return f"an image of size {width}x{height}."
    except Exception as e:
        print(f"Error processing image for description: {e}")
        return "an unreadable image."


def transcribe_audio(audio_bytes):
    """Transcribes audio bytes to text using SpeechRecognition."""
    r = sr.Recognizer()
    try:
        # Use soundfile to read audio data as SpeechRecognition can be particular about formats
        with sf.SoundFile(io.BytesIO(audio_bytes), 'r') as f:
            data = f.read(dtype='int16')
            samplerate = f.samplerate


        audio_data = sr.AudioData(data.tobytes(), samplerate, 2)  # 2 bytes per sample for int16 (int16 is 2 bytes)

        # Using Google Web Speech API (requires internet connection)
        text = r.recognize_google(audio_data)
        print(f"Transcribed audio: '{text}'")
        return text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None


# --- 3. Domain and Intent Recognition ---
# This function is now more robust to consider previous multimodal context
def identify_domain_and_intent(user_input, has_image=False, has_voice=False):
    """Identifies the domain and intent of the user's query, considering ongoing multimodal context."""
    user_input_lower = user_input.lower()

    # Prioritize current turn multimodal input
    if has_image:
        st.session_state.last_uploaded_image_bytes = st.session_state.current_uploaded_image_bytes  # Store for next turn context
        st.session_state.last_uploaded_audio_bytes = None  # Clear audio context
        return "image_query", "visual_question"
    if has_voice:
        st.session_state.last_uploaded_audio_bytes = st.session_state.current_uploaded_audio_bytes  # Store for next turn context
        st.session_state.last_uploaded_image_bytes = None  # Clear image context
        # If voice input, the content will be processed as text after transcription.
        # So we should still run text NLU on the transcribed text for actual intent.
        return "voice_query", "audio_input"

    # If no new multimodal input this turn, check for lingering multimodal context from previous turn
    # And if the user's current text query seems to be about the previous multimodal input
    if st.session_state.get('last_uploaded_image_bytes') and \
            any(kw in user_input_lower for kw in ["what", "describe", "identify", "picture", "image", "this", "it"]):
        return "image_query", "visual_question"

    if st.session_state.get('last_uploaded_audio_bytes') and \
            any(kw in user_input_lower for kw in ["what did i say", "transcribe", "audio", "voice", "what was that"]):
        return "voice_query", "audio_input"  # User is asking about previous audio

    # Keyword-based intent recognition for specific text-based domains
    if any(keyword in user_input_lower for keyword in
           ["product", "item", "cost", "buy", "price", "warranty", "shipping", "order"]):
        if "product a" in user_input_lower: return "product_support", "product_A_info"
        if "product b" in user_input_lower: return "product_support", "product_B_info"
        if "warranty" in user_input_lower: return "product_support", "warranty_info"
        if "shipping" in user_input_lower: return "product_support", "shipping_info"
        return "product_support", "general_product_inquiry"
    elif any(keyword in user_input_lower for keyword in
             ["technical", "support", "troubleshoot", "fix", "error", "problem", "reset", "software", "internet",
              "network", "password"]):
        if "internet" in user_input_lower or "network" in user_input_lower: return "technical_support", "internet_troubleshoot"
        if "password" in user_input_lower: return "technical_support", "password_reset"
        if "software" in user_input_lower: return "technical_support", "software_issue"
        return "technical_support", "general_technical_inquiry"
    elif any(keyword in user_input_lower for keyword in ["joke", "tell me a joke", "funny"]):
        return "general", "joke"
    elif any(keyword in user_input_lower for keyword in ["hello", "hi", "hey", "greetings"]):
        return "general", "greeting"
    elif any(keyword in user_input_lower for keyword in ["bye", "goodbye", "farewell", "exit", "thanks bye"]):
        return "general", "farewell"
    elif any(
            keyword in user_input_lower for keyword in ["what can you do", "help me", "your capabilities", "features"]):
        return "general", "capabilities"
    else:
        return "general", "default"


# --- 4. Response Generation Logic ---
# This function now correctly uses the last_uploaded_bytes from session state
def generate_chatbot_response(
        general_chat_pipeline, vqa_pipeline,
        domain, intent, user_input, chat_history,
        transcribed_audio_text=None
):
    response = ""

    # Determine which multimodal data to use: current turn's or last turn's context
    actual_image_bytes = st.session_state.current_uploaded_image_bytes or st.session_state.get(
        'last_uploaded_image_bytes')
    actual_audio_bytes = st.session_state.current_uploaded_audio_bytes or st.session_state.get(
        'last_uploaded_audio_bytes')

    # --- Prioritize multimodal specific handling based on identified domain/intent ---
    if domain == "image_query":
        if not actual_image_bytes:  # If image_query domain but no image present (e.g., user just typed "about the picture")
            response = "Please upload an image first so I can answer questions about it!"
        elif vqa_pipeline:
            try:
                image_pil = Image.open(io.BytesIO(actual_image_bytes))
                vqa_result = vqa_pipeline(image=image_pil, question=user_input)
                # Ensure the VQA result is not empty before using it
                response = vqa_result[0]['answer'] if vqa_result and vqa_result[0]['answer'] else \
                    f"I analyzed the image for your question '{user_input}', but couldn't find a specific answer using my visual recognition. Can you try rephrasing?"
                print(f"VQA model generated: {response}")
            except Exception as e:
                print(f"Error during VQA model inference: {e}. Simulating response.")
                image_processed_desc = process_image_data_for_desc(actual_image_bytes)
                response = DOMAIN_KNOWLEDGE_BASES["image_query"]["simulate_vqa_response"](user_input,
                                                                                          image_processed_desc)
        else:
            # Fallback if VQA pipeline wasn't loaded or specified
            image_processed_desc = process_image_data_for_desc(actual_image_bytes)
            response = DOMAIN_KNOWLEDGE_BASES["image_query"]["simulate_vqa_response"](user_input, image_processed_desc)

    elif domain == "voice_query":
        if not actual_audio_bytes:
            response = "Please upload an audio file first, or speak your query if you want me to transcribe."
        elif transcribed_audio_text:  # This means transcription was successful this turn or in the last turn
            response = DOMAIN_KNOWLEDGE_BASES["voice_query"]["simulate_voice_response"](transcribed_audio_text)
            # For even better behavior, you could re-route the transcribed_audio_text
            # through 'identify_domain_and_intent' again to get its true text-based intent
            # and then generate a text response for that intent here.
            # (Example commented out for current complexity)
            # text_domain, text_intent = identify_domain_and_intent(transcribed_audio_text)
            # response_from_transcribed = "Based on your voice input, you asked: " + transcribed_audio_text + "\n" + \
            #                            generate_chatbot_response(general_chat_pipeline, vqa_pipeline, text_domain, text_intent, transcribed_audio_text, chat_history)
            # response = response_from_transcribed
        else:
            response = DOMAIN_KNOWLEDGE_BASES["voice_query"]["unidentified_voice"]  # Fallback if transcription failed

    # Handle text-based domains based on identified intent
    elif domain == "product_support":
        if intent == "product_A_info":
            response = DOMAIN_KNOWLEDGE_BASES["product_support"]["product_A"]
        elif intent == "product_B_info":
            response = DOMAIN_KNOWLEDGE_BASES["product_support"]["product_B"]
        elif intent == "warranty_info":
            response = DOMAIN_KNOWLEDGE_BASES["product_support"]["warranty"]
        elif intent == "shipping_info":
            response = DOMAIN_KNOWLEDGE_BASES["product_support"]["shipping"]
        else:
            response = DOMAIN_KNOWLEDGE_BASES["product_support"]["product_info"]
    elif domain == "technical_support":
        if intent == "internet_troubleshoot":
            response = DOMAIN_KNOWLEDGE_BASES["technical_support"]["troubleshoot_internet"]
        elif intent == "password_reset":
            response = DOMAIN_KNOWLEDGE_BASES["technical_support"]["reset_password"]
        elif intent == "software_issue":
            response = DOMAIN_KNOWLEDGE_BASES["technical_support"]["software_issue"]
        else:
            response = DOMAIN_KNOWLEDGE_BASES["technical_support"]["common_issues"]
    elif domain == "general":
        if intent == "greeting":
            response = DOMAIN_KNOWLEDGE_BASES["general"]["greeting"]
        elif intent == "farewell":
            response = DOMAIN_KNOWLEDGE_BASES["general"]["farewell"]
        elif intent == "capabilities":
            response = DOMAIN_KNOWLEDGE_BASES["general"]["capabilities"]
        elif intent == "joke":
            response = DOMAIN_KNOWLEDGE_BASES["general"]["joke"]
        else:  # This block handles the "any type of question" via LLM (DistilGPT2)
            if general_chat_pipeline:
                try:
                    # Construct chat history for context for the LLM
                    # Include a few recent user and chatbot messages for better conversational context
                    context_messages = []
                    # Limit to last 4 interactions to keep the prompt manageable for DistilGPT2
                    for msg in reversed(chat_history):
                        if len(context_messages) >= 4:
                            break
                        if msg['role'] == 'user':
                            context_messages.insert(0, f"User: {msg['message']}")
                        elif msg['role'] == 'assistant':
                            context_messages.insert(0, f"PyChat: {msg['message']}")  # Changed to PyChat
                    context_str = "\n".join(context_messages)

                    # Final prompt for the LLM, clearly indicating whose turn it is
                    full_input_text = f"{context_str.strip()}\nUser: {user_input}\nPyChat:"  # Changed to PyChat

                    # Debugging: Print the exact prompt sent to the LLM
                    print(f"\n--- LLM Prompt Start ---\n{full_input_text}\n--- LLM Prompt End ---")

                    generated_output = general_chat_pipeline(
                        full_input_text,
                        **GENERATION_PARAMS,
                        pad_token_id=general_chat_pipeline.tokenizer.eos_token_id,  # Essential for stopping generation
                    )
                    raw_generated_text = generated_output[0]['generated_text']

                    # Debugging: Print the raw output from the LLM
                    print(f"\n--- Raw LLM Output Start ---\n{raw_generated_text}\n--- Raw LLM Output End ---")

                    # --- Improved Post-Processing for Extraction and Repetition Filtering ---
                    response = raw_generated_text

                    # 1. Strip the *entire* prompt from the raw output first
                    if response.startswith(full_input_text):
                        response = response[len(full_input_text):].strip()

                    # 2. Further strip any common LLM markers
                    # Added 'PyChat:' to the list of strings to strip
                    response = response.replace("Chatbot:", "").replace("User:", "").replace("Customer:", "").replace(
                        "PyChat:", "").strip()

                    # 3. Clean up common conversational filler (add more as you observe)
                    common_fillers = [
                        "Hello!", "How can I assist you today?", "Hi there!", "I'm really sorry,",
                        "I'm a newbie", "do you remember the time", "the guy who was the guy",
                        "I've been on a lot of the games lately.", "I've had quite a few games,",
                        "I'm happy to play them.", "I'm pretty happy to play them all.",
                        "hey, now I'm trying to help you.", "good, good, good,", "How can I help you today?"
                    ]
                    for filler in common_fillers:
                        response = response.replace(filler, "").strip()  # Remove exact filler
                        response = response.replace(filler.lower(), "").strip()  # Remove lowercase filler

                    # 4. Final check for repetition or emptiness, comparing against original user input
                    user_input_lower = user_input.lower().strip()
                    response_lower = response.lower().strip()

                    if (not response or len(response) < 5 or  # Response is empty or too short
                            "i'm not sure" in response_lower or  # Generic "I don't know" phrase
                            response_lower == user_input_lower or  # Exact echo of the user's input
                            # Check if user input is contained AND response is not significantly longer (indicates minor echo)
                            (user_input_lower in response_lower and len(response) < 2 * len(user_input) and len(
                                user_input) > 5)
                    ):
                        response = DOMAIN_KNOWLEDGE_BASES["general"]["default"]

                    response = response.strip()

                except Exception as e:
                    print(f"Error during text generation with general chat pipeline: {e}. Falling back to default.")
                    response = DOMAIN_KNOWLEDGE_BASES["general"]["default"]
            else:
                response = DOMAIN_KNOWLEDGE_BASES["general"]["default"]
    else:
        # Fallback for any unhandled domain/scenario (should ideally not be reached if identify_domain_and_intent is robust)
        response = DOMAIN_KNOWLEDGE_BASES["general"]["default"]

    return response


# --- Streamlit UI Layout and Main Logic ---
# Updated page_title to reflect the chatbot name
st.set_page_config(page_title="PyChat - Your AI Companion", page_icon="🤖", layout="centered")

# Changed the title to be centered using HTML and Markdown
st.markdown("<h1 style='text-align: center;'>🤖 PyChat</h1>", unsafe_allow_html=True)

# Made the subline a bigger font size using a Markdown heading (###)
st.markdown(
    "### **Your friendly AI companion, here to help you navigate information with ease and a smile!**"
)
st.markdown(
    "I can help with product inquiries, technical support, and general questions. You can also upload images and voice recordings and ask questions about them!"
)
st.markdown("Type `clear` to clear the conversation and restart.")

# Initialize Streamlit Session State for chat history and context
# These variables persist across reruns of the script
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_domain" not in st.session_state:
    st.session_state.current_domain = "general"
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "transcribed_audio" not in st.session_state:  # Stores the last successfully transcribed text
    st.session_state.transcribed_audio = None
# --- NEW: Session state to retain multimodal context across turns ---
if 'last_uploaded_image_bytes' not in st.session_state:
    st.session_state.last_uploaded_image_bytes = None
if 'last_uploaded_audio_bytes' not in st.session_state:
    st.session_state.last_uploaded_audio_bytes = None
# Current turn's raw uploaded bytes (these will be transferred to 'last_uploaded_bytes' if processed)
st.session_state.current_uploaded_image_bytes = None
st.session_state.current_uploaded_audio_bytes = None

# Add keys to the file uploader widgets to enable clearing their state
if 'image_uploader_key' not in st.session_state:
    st.session_state.image_uploader_key = 0
if 'audio_uploader_key' not in st.session_state:
    st.session_state.audio_uploader_key = 0


# Function to clear file uploaders by incrementing their key and clearing context
def clear_file_uploaders():
    st.session_state.image_uploader_key += 1
    st.session_state.audio_uploader_key += 1
    st.session_state.last_uploaded_image_bytes = None  # Clear context on explicit clear
    st.session_state.last_uploaded_audio_bytes = None  # Clear context on explicit clear


# Display chat messages from history
# Iterates through the chat_history and displays each message
for message_obj in st.session_state.chat_history:
    with st.chat_message(message_obj["role"]):  # Uses Streamlit's chat message container
        st.write(message_obj["message"])  # Display the text message
        if message_obj["image"]:
            st.image(message_obj["image"], caption=f"{message_obj['role']} Image", width=200)
        if message_obj["audio_bytes"]:
            # Display an audio player for uploaded voice
            st.audio(message_obj["audio_bytes"], format='audio/wav')

# Input Components for user interaction
user_query = st.chat_input("Ask PyChat a question...")  # Updated placeholder text!
st.sidebar.header("Upload Files")
# File uploader for images - now with a dynamic key and setting current_uploaded_image_bytes
uploaded_image_file = st.sidebar.file_uploader(
    "Upload an image (PNG, JPG, JPEG)",
    type=["png", "jpg", "jpeg"],
    key=f"image_uploader_{st.session_state.image_uploader_key}"  # Dynamic key
)
if uploaded_image_file:
    st.session_state.current_uploaded_image_bytes = uploaded_image_file.getvalue()
else:
    st.session_state.current_uploaded_image_bytes = None  # Explicitly set to None if no file uploaded this turn

# File uploader for voice - now with a dynamic key and setting current_uploaded_audio_bytes
uploaded_audio_file = st.sidebar.file_uploader(
    "Upload voice (WAV, MP3, FLAC)",
    type=["wav", "mp3", "flac"],
    key=f"audio_uploader_{st.session_state.audio_uploader_key}"  # Dynamic key
)
if uploaded_audio_file:
    st.session_state.current_uploaded_audio_bytes = uploaded_audio_file.getvalue()
else:
    st.session_state.current_uploaded_audio_bytes = None  # Explicitly set to None if no file uploaded this turn

# --- Main processing logic when user interacts (either by typing or uploading a file) ---
# Process input only if there's actual new input from text or current file upload
if user_query or st.session_state.current_uploaded_image_bytes or st.session_state.current_uploaded_audio_bytes:
    current_user_input = user_query if user_query else ""
    has_image_input_this_turn = st.session_state.current_uploaded_image_bytes is not None
    has_audio_input_this_turn = st.session_state.current_uploaded_audio_bytes is not None
    transcribed_text = None  # Will hold the text from audio if applicable

    # Handle 'clear' command first and explicitly. This must be the very first check for user_query.
    if current_user_input.lower().strip() == 'clear':
        st.session_state.chat_history = []
        st.session_state.current_domain = "general"
        st.session_state.last_intent = None
        st.session_state.transcribed_audio = None
        clear_file_uploaders()  # Clear the uploaders on 'clear' command, also clears last_uploaded_bytes
        st.rerun()  # Rerun to clear the UI and state immediately

    # --- 1. Handle Audio Transcription (if any) ---
    if has_audio_input_this_turn:
        with st.spinner("Transcribing audio..."):
            transcribed_text = transcribe_audio(st.session_state.current_uploaded_audio_bytes)
            if transcribed_text:
                st.session_state.transcribed_audio = transcribed_text
                # If the user only uploaded audio and didn't type text, use transcribed text as the main input
                if not current_user_input:
                    current_user_input = transcribed_text
            else:
                st.sidebar.warning("Could not transcribe audio. Please try again or type your query.")
                # If transcription fails, the chatbot will proceed with whatever (if anything) was typed or default behavior.

    # --- 2. Determine Primary Interaction Type for Routing ---
    # Now, identify_domain_and_intent uses the *current* turn's direct upload flags
    # and *also* the session state for last_uploaded_bytes for follow-up questions.
    primary_domain, primary_intent = identify_domain_and_intent(
        current_user_input,
        has_image=has_image_input_this_turn,
        has_voice=has_audio_input_this_turn
    )

    # --- 3. Add user message to history (for display) ---
    st.session_state.chat_history.append({
        "role": "user",
        "message": current_user_input if current_user_input else "_(Image/Audio Uploaded)_",
        "image": st.session_state.current_uploaded_image_bytes,  # Store current bytes for display
        "audio_bytes": st.session_state.current_uploaded_audio_bytes  # Store current bytes for display
    })

    # --- 4. Generate Chatbot Response ---
    st.session_state.current_domain = primary_domain  # Update session state after robust routing
    st.session_state.last_intent = primary_intent

    with st.spinner(f"PyChat is thinking in {primary_domain} domain..."):  # Changed to PyChat
        chatbot_response = generate_chatbot_response(
            GENERAL_CHAT_PIPELINE,
            VQA_PIPELINE,
            primary_domain,  # Pass the robustly determined domain
            primary_intent,  # Pass the robustly determined intent
            current_user_input,
            st.session_state.chat_history,
            transcribed_audio_text=st.session_state.transcribed_audio  # Always pass the last known transcribed text
        )

    # --- 5. Add chatbot response to history ---
    st.session_state.chat_history.append({
        "role": "assistant",
        "message": chatbot_response,
        "image": None,  # Chatbot does not generate images currently
        "audio_bytes": None  # Chatbot does not generate audio currently
    })

    # --- 6. Clear current turn's file data and rerun to update the UI ---
    # The last_uploaded_bytes for context are handled within identify_domain_and_intent
    # and only cleared on explicit 'clear' command or new file upload.
    clear_file_uploaders()  # Call function to clear uploaders' UI state and context
    st.rerun()  # This clears the input fields and refreshes the chat display
