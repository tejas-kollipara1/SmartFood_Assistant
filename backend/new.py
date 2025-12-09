import os
import sys
import asyncio
import pygame
import requests
import edge_tts
import sounddevice as sd
import soundfile as sf

from openai import OpenAI

# ================== CONFIG ==================

# Read OpenAI key ONLY from environment. No default, no hardcoding.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please set it in your environment before running."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# Chat + STT models
CHAT_MODEL = "gpt-4.1-mini"              # text brain
TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"  # speech-to-text (non-Whisper)

VOICE_MAP = {
    "hi": "hi-IN-SwaraNeural",
    "te": "te-IN-MohanNeural",
    "ta": "ta-IN-PallaviNeural",
    "kn": "kn-IN-GaganNeural",
    "bn": "bn-IN-TanishaaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "en": "en-IN-NeerjaNeural",
}

# ================== AUDIO HELPERS ==================


def record_audio(filename, duration=10):
    """
    Records audio using the default input device.
    """
    print(f"\nRecording for {duration} seconds... (Speak Now!)")

    fs = 44100
    device_id = None  # let sounddevice choose default mic

    try:
        recording = sd.rec(
            int(duration * fs),
            samplerate=fs,
            channels=1,
            device=device_id,
            blocking=True,
        )
        sf.write(filename, recording, fs)
        print(f"Recording saved to {filename}")

    except Exception as e:
        print(f"\nMicrophone Error: {e}")
        print("Check your audio input device (mic settings).")


async def speak(text, lang_code):
    """
    Uses EdgeTTS to speak the text in the correct language.
    """
    if not text:
        return

    print(f"\nAI Speaking ({lang_code}): {text}")

    voice = VOICE_MAP.get(lang_code, "en-IN-NeerjaNeural")
    output_file = "ai_response.mp3"

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_file)

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()

        os.remove(output_file)
    except Exception as e:
        print(f"Audio Playback Error: {e}")


def generate_image(prompt, filename="final_dish.jpg"):
    """
    Generate an image using Pollinations.ai and save locally.
    """
    print(f"\n[Pollinations] Generating image for: {prompt!r}")

    try:
        encoded_prompt = requests.utils.quote(prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            print(f"[Pollinations] Failed with status {r.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        print(f"[Pollinations] Image saved to {filename}")

        try:
            if sys.platform.startswith("win"):
                os.startfile(filename)
            elif sys.platform.startswith("darwin"):
                os.system(f'open "{filename}"')
            else:
                os.system(f'xdg-open "{filename}"')
        except Exception as e:
            print("[Pollinations] Could not auto-open image:", e)

    except Exception as e:
        print("\n[Pollinations ERROR] Unexpected error while generating image.")
        print(type(e).__name__, ":", e)


# ================== OPENAI HELPERS ==================


def transcribe_audio(filename: str) -> str:
    """
    Uses OpenAI gpt-4o-mini-transcribe to convert speech -> text.
    (This is your 'Whisper role', but it's not whisper-1.)
    """
    try:
        with open(filename, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=TRANSCRIBE_MODEL,
                file=f,
            )
        text = transcript.text.strip()
        print(f"\n[Transcription from {filename}]: {text}")
        return text
    except Exception as e:
        print(f"\n[Transcription ERROR] {e}")
        return ""


def chat(prompt: str) -> str:
    """
    Simple helper to call OpenAI chat model and return plain text.
    """
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"\n[Chat ERROR] {e}")
        return ""


# ================== PARSERS (UNCHANGED LOGIC) ==================


def parse_lang_and_options(response_text):
    lang_code = "en"
    ai_question = response_text

    for line in response_text.split("\n"):
        if "LANG:" in line:
            lang_code = line.split(":", 1)[1].strip().lower()
        if "OPTIONS:" in line:
            ai_question = line.split(":", 1)[1].strip()

    return lang_code, ai_question


def parse_dish_block(final_text):
    dish_name = ""
    ingredients_text = ""
    question_text = ""
    base_recipe_text = ""
    image_prompt = "Indian food"

    for line in final_text.split("\n"):
        line = line.strip()
        if line.startswith("DISH:"):
            dish_name = line.split(":", 1)[1].strip()
        elif line.startswith("INGREDIENTS:"):
            ingredients_text = line.split(":", 1)[1].strip()
        elif line.startswith("QUESTION:"):
            question_text = line.split(":", 1)[1].strip()
        elif line.startswith("RECIPE:"):
            base_recipe_text = line.split(":", 1)[1].strip()
        elif line.startswith("IMG:"):
            image_prompt = line.split(":", 1)[1].strip()

    return dish_name, ingredients_text, question_text, base_recipe_text, image_prompt


def parse_steps(steps_text):
    """
    Parses lines like:
      STEP 1: ...
      STEP 2: ...
    into a Python list of step strings.
    """
    steps = []
    for line in steps_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("step"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                steps.append(parts[1].strip())
            else:
                parts = line.split("-", 1)
                if len(parts) == 2:
                    steps.append(parts[1].strip())
    return steps


def get_user_step_command(lang_code, step_index):
    """
    Records the user's voice command for the current step,
    uses OpenAI STT + chat to interpret it as NEXT / REPEAT / STOP.
    """
    cmd_file = f"step_command_{step_index}.wav"
    input("\nPress Enter and then SAY 'next', 'repeat', or 'stop' (in your language)...")
    record_audio(cmd_file, duration=5)

    cmd_text = transcribe_audio(cmd_file)

    prompt_cmd = f"""
You are a controller for a cooking assistant.

Language code: {lang_code}.
User just spoke this (transcribed from audio):
\"\"\"{cmd_text}\"\"\"

Your job: Map their intent to exactly one of these commands:
NEXT, REPEAT, STOP

Rules:
- If they want to go to the next step, answer: NEXT
- If they want you to repeat the current step, answer: REPEAT
- If they want to finish or stop, answer: STOP

Respond with ONLY ONE WORD in UPPERCASE: NEXT or REPEAT or STOP.
Do NOT include any other text.
"""

    command_raw = chat(prompt_cmd).upper()
    print(f"\n--- OpenAI Step Command Raw ---\n{command_raw}\n")

    if "REPEAT" in command_raw:
        return "REPEAT"
    if "STOP" in command_raw:
        return "STOP"
    return "NEXT"


# ================== MAIN FLOW ==================


def main():
    print("SMART CHEF AI (OpenAI version): Starting...")

    # --- STEP 1: USER SPEAKS INGREDIENTS ---
    ingredients_file = "ingredients.wav"
    input("\nPress Enter to RECORD INGREDIENTS (5 seconds)...")
    record_audio(ingredients_file, duration=5)

    ingredients_text_heard = transcribe_audio(ingredients_file)

    print("Asking OpenAI to detect language and suggest 2 dishes...")

    prompt_1 = f"""
You are a multilingual Indian cooking assistant.

User described their ingredients in an Indian language (or English).
Here is the transcription of what they said:
\"\"\"{ingredients_text_heard}\"\"\"


1. Identify the Indian language spoken (return code: hi, te, ta, kn, bn, ml, en).
2. In that SAME language, suggest 2 distinct dishes based on these ingredients.
3. Return ONLY this format:
   LANG: [code]
   OPTIONS: [Your question with the two dish options, asking the user to choose one]
"""

    response_text = chat(prompt_1)

    print("\n--- OpenAI Response 1 ---")
    print(response_text)

    lang_code, ai_question = parse_lang_and_options(response_text)

    asyncio.run(speak(ai_question, lang_code))

    # --- STEP 2: USER SPEAKS CHOICE OF DISH ---
    choice_file = "choice.wav"
    input("\nPress Enter to RECORD YOUR CHOICE (up to 10 seconds)...")
    record_audio(choice_file, duration=10)

    choice_text = transcribe_audio(choice_file)

    print("Understanding your chosen dish and preparing base recipe...")

    prompt_2 = f"""
You are a multilingual Indian cooking assistant.

Earlier, you asked the user this question (with 2 dish options) in language {lang_code}:
\"\"\"{ai_question}\"\"\"

Now, the user replied with this (transcribed from audio):
\"\"\"{choice_text}\"\"\"


Tasks:
1. Decide which ONE dish (from your earlier two options) the user selected.
2. In the SAME language ({lang_code}), respond in EXACTLY this format:
   DISH: [name of the chosen dish in {lang_code}]
   INGREDIENTS: [a clear comma-separated list of ingredients in {lang_code}]
   QUESTION: [politely ask the user if they have all these ingredients; tell them to say which are missing, or say 'yes' if everything is available]
   RECIPE: [2 short sentences describing how to cook it in {lang_code}]
   IMG: [English name of the dish]

Important:
- Do NOT add any extra lines or text outside this format.
"""

    final_text = chat(prompt_2)
    print("\n--- OpenAI Response 2 ---")
    print(final_text)

    dish_name, ingredients_text, question_text, base_recipe_text, image_prompt = parse_dish_block(
        final_text
    )

    generate_image(f"delicious {image_prompt}, professional food photography, 4k")

    # --- STEP 3: ASK USER IF THEY HAVE ALL INGREDIENTS ---
    ingredients_prompt_for_user = f"{dish_name}. {ingredients_text}. {question_text}"
    asyncio.run(speak(ingredients_prompt_for_user, lang_code))

    ingredients_reply_file = "ingredients_reply.wav"
    input(
        "\nPress Enter to RECORD YOUR INGREDIENTS AVAILABILITY "
        "(say which items you don't have, or say you have all) (up to 10 seconds)..."
    )
    record_audio(ingredients_reply_file, duration=10)

    ingredients_reply_text = transcribe_audio(ingredients_reply_file)

    print("Adjusting recipe based on your available ingredients...")

    prompt_3 = f"""
You are a helpful cooking assistant.

Language Code: {lang_code}
The chosen dish is: {dish_name}

Here is the original ingredient list (in {lang_code}):
{ingredients_text}

Here is a short base recipe (in {lang_code}):
{base_recipe_text}

Now here is what the user said about which ingredients they have or don't have
(transcribed from their audio, in {lang_code} or mixed):
\"\"\"{ingredients_reply_text}\"\"\"


Your tasks:
1. Understand whether:
   - The user has all ingredients, OR
   - Some ingredients are missing and which ones.
2. If the user has all ingredients:
   - Keep the same dish and provide a clear final recipe in {lang_code},
     3–5 sentences, step-by-step.
3. If the user is missing some ingredients:
   - Decide if the same dish can still be made without those ingredients.
     If YES: adjust the recipe accordingly and clearly mention any substitutions or skips.
     If NO: suggest a simple alternative dish that can be made from the ingredients they likely have,
     and give a 3–5 sentence recipe for that alternative.
4. IMPORTANT: Respond ONLY with the final recipe text in {lang_code}.
   Do NOT include labels like DISH:, INGREDIENTS:, IMG:, or any English explanations.
"""

    final_recipe_text = chat(prompt_3)
    print("\n--- OpenAI Response 3 (Final Recipe) ---")
    print(final_recipe_text)

    asyncio.run(speak(final_recipe_text, lang_code))

    # --- STEP 6: STEP-BY-STEP ---
    print("\nAsking OpenAI to convert recipe into step-by-step instructions...")

    prompt_steps = f"""
Language code: {lang_code}

Here is the final recipe text in {lang_code}:
\"\"\"{final_recipe_text}\"\"\"


Convert this into a clear numbered sequence of short steps in {lang_code}.
Respond in EXACTLY this format:
STEP 1: ...
STEP 2: ...
STEP 3: ...
(and so on)

Do NOT add any introduction or conclusion.
Do NOT add any text that does not start with 'STEP'.
"""

    steps_text = chat(prompt_steps)
    print("\n--- OpenAI Step List ---")
    print(steps_text)

    steps = parse_steps(steps_text)

    if not steps:
        print("\nCould not parse steps. Using full recipe only.")
        return

    intro_prompt = f"""
Language code: {lang_code}.
Write one short sentence in this language telling the user:
'Now we will go through the recipe step by step. After each step, say NEXT to continue, REPEAT to hear it again, or STOP to finish.'
Respond only with that sentence, in {lang_code}.
"""
    intro_text = chat(intro_prompt)
    asyncio.run(speak(intro_text, lang_code))

    print("\nEntering interactive step-by-step mode...")

    step_index = 0
    while step_index < len(steps):
        current_step = steps[step_index]
        print(f"\nSTEP {step_index + 1}: {current_step}")
        asyncio.run(speak(f"Step {step_index + 1}: {current_step}", lang_code))

        command = get_user_step_command(lang_code, step_index + 1)
        print(f"Interpreted command: {command}")

        if command == "NEXT":
            step_index += 1
        elif command == "REPEAT":
            continue
        elif command == "STOP":
            print("\nUser stopped the recipe.")
            break
        else:
            step_index += 1

    print("\nCooking assistant finished. Bon Appétit!")


if __name__ == "__main__":
    main()
