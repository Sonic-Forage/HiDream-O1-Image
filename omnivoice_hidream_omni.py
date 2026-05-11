"""
MindForge Omni v2 — Full Creative AI Stack.

Three services:
  - QwenBrain (CPU) — Qwen3.6-27B via OpenRouter API: creative writing, prompt engineering
  - OmniVoiceRunner (A10G) — TTS with MindExpander's cloned voice
  - HiDreamRunner (A10G) — Image generation up to 2048x2048

Endpoints:
  GET  /health                           — health check
  POST /v1/audio/speech                  — TTS
  POST /v1/images/generations            — Image gen (with auto prompt enhancement)
  POST /v1/omni/create                   — Full creative: brain → voice + image
  POST /v1/brain/chat                    — Direct Qwen3.6 chat

Usage:
  modal deploy omnivoice_hidream_omni.py
"""

import modal

APP_NAME = "mindforge-omni"
MODEL_VOLUME = "omnivoice-outputs"
HF_CACHE_VOLUME = "huggingface-cache"

# ── Base images ───────────────────────────────────────────────────────
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "sox", "espeak-ng",
                 "portaudio19-dev", "libgl1-mesa-glx", "libglib2.0-0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface"})
)

# TTS image — bake OmniVoice repo into image so no clone on cold start
tts_image = (
    base_image
    .pip_install("torch==2.8.0", "torchaudio==2.8.0",
                 extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers>=5.3.0", "accelerate", "webdataset", "soundfile",
                 "librosa", "pydub", "tensorboardX", "numpy", "datasets",
                 "huggingface_hub", "hf_transfer", "safetensors", "einops",
                 "snac", "descript-audio-codec")
    .add_local_dir("/opt/data/workspace/M1ND3XPAND3RS-VOICE-VoxCPM-ready",
                   remote_path="/dataset", copy=True)
    # Pre-clone OmniVoice repo — saves ~15s on every cold start
    .run_commands("git clone --depth 1 https://github.com/k2-fsa/OmniVoice.git /workspace/OmniVoice")
)

# Image gen image — bake HiDream repo into image
image_gen_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0",
                 extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install("transformers==4.57.1", "diffusers", "accelerate", "einops",
                 "numpy", "pillow", "tqdm", "scipy", "flask", "openai",
                 "huggingface_hub", "hf_transfer", "safetensors", "qwen-vl-utils")
    .run_commands("pip install flash-attn --no-build-isolation")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/huggingface"})
    # Pre-clone HiDream repo — saves ~10s on every cold start
    .run_commands("git clone --depth 1 https://github.com/HiDream-ai/HiDream-O1-Image.git /workspace/HiDream-O1-Image")
)

# Brain image — just needs OpenAI client for API calls
brain_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("openai", "httpx")
)

# CPU ASGI
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi", "uvicorn", "httpx")
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")
openrouter_secret = modal.Secret.from_name("openrouter-secret")


# ══════════════════════════════════════════════════════════════════════
# CPU Class 1: QwenBrain — Qwen3.6-27B via OpenRouter API
# ══════════════════════════════════════════════════════════════════════
@app.cls(
    image=brain_image,
    cpu=0.5,
    memory=512,
    secrets=[openrouter_secret],
    scaledown_window=60,
    timeout=120,
    max_containers=1,
)
@modal.concurrent(max_inputs=8)
class QwenBrain:
    """Qwen3.6-27B via OpenRouter — creative writing, prompt engineering."""

    @modal.enter()
    def setup(self):
        import os
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.model = "qwen/qwen3.6-27b"
        print(f"✅ Brain connected: {self.model} via OpenRouter")

    @modal.method()
    def chat(self, messages: list, max_tokens: int = 2048, temperature: float = 0.7) -> str:
        import re
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"enable_thinking": False},
        )
        content = response.choices[0].message.content or ""
        # Strip thinking tags
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        # Strip "Thinking Process:" blocks (Qwen3.6 without tags)
        content = re.sub(r"Thinking Process:.*?(?=\n\n[A-Z]|\n\n\*\*|\n\n[^\*])", "", content, flags=re.DOTALL).strip()
        # Strip numbered reasoning steps
        content = re.sub(r"^\d+\.\s+\*\*.*?\*\*:.*?\n", "", content, flags=re.MULTILINE).strip()
        # If still mostly thinking, try to extract the actual content
        if len(content) > 500 and any(w in content.lower() for w in ["brainstorm", "draft", "attempt", "syllable"]):
            lines = content.split("\n")
            creative = [l for l in lines if l.strip() and len(l.strip()) < 80
                        and not l.strip().startswith(("*", "-", "1.", "2.", "3.", "4.", "5."))
                        and not any(w in l.lower() for w in ["brainstorm", "draft", "attempt", "syllable", "analyze", "deconstruct"])]
            if creative:
                content = "\n".join(creative[-10:])
        return content

    @modal.method()
    def write_poem(self, theme: str, style: str = "free verse") -> dict:
        """Write a poem and generate an optimized image prompt."""
        import json, re
        result = self.chat.local([
            {"role": "system", "content": """You are a creative AI that writes poems and generates image prompts.
Given a theme and style, respond with ONLY a JSON object:
{
  "poem": "the poem text",
  "title": "poem title",
  "image_prompt": "detailed English prompt for image generation, 80-200 words"
}

The image_prompt should be a detailed, self-contained description for an AI image generator.
Include: subject, composition, action, location, image style, camera/lens/lighting specs.
Make it cinematic, vivid, and specific. Use full sentences, not keywords."""},
            {"role": "user", "content": f"Theme: {theme}\nStyle: {style}"},
        ], temperature=0.9)

        text = result.strip()
        try:
            if "```" in text:
                m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception:
            pass
        return {
            "poem": result,
            "title": theme,
            "image_prompt": f"cinematic illustration of {style}, {theme}, highly detailed, dramatic lighting, 8k masterpiece",
        }

    @modal.method()
    def enhance_prompt(self, user_prompt: str) -> str:
        """Enhance a user prompt into an optimized image generation prompt."""
        return self.chat.local([
            {"role": "system", "content": """You are an expert AI image prompt engineer.
Rewrite the user's request into a detailed, self-contained English prompt for image generation.
Include: subject details, composition, lighting, camera angle, art style, color palette.
Output ONLY the enhanced prompt — no explanations, no JSON, just the prompt text.
Length: 80-200 words. Use full sentences."""},
            {"role": "user", "content": user_prompt},
        ], max_tokens=512, temperature=0.7).strip()


# ══════════════════════════════════════════════════════════════════════
# GPU Class 2: OmniVoice TTS (A10G)
# ══════════════════════════════════════════════════════════════════════
@app.cls(
    image=tts_image,
    gpu="A10G",
    volumes={"/outputs": vol, "/cache": hf_cache},
    secrets=[hf_secret],
    scaledown_window=120,
    timeout=600,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
class OmniVoiceRunner:
    """MindExpander's cloned voice — TTS endpoint."""

    @modal.enter()
    def load_model(self):
        import os, sys

        # OmniVoice repo is pre-baked into the image at /workspace/OmniVoice
        repo_dir = "/workspace/OmniVoice"
        os.environ["PYTHONPATH"] = f"{repo_dir}:{os.environ.get('PYTHONPATH', '')}"
        sys.path.insert(0, repo_dir)

        vol.reload()
        model_path = "/outputs/checkpoints/checkpoint-1500"
        if not os.path.exists(model_path):
            model_path = "TheMindExpansionNetwork/omnivoice-mindexpander-voice"
        print(f"🎤 Loading OmniVoice from {model_path}...")
        from omnivoice.models.omnivoice import OmniVoice
        self.model = OmniVoice.from_pretrained(model_path).to("cuda").eval()
        self.model_path = model_path
        self.default_ref_audio = "/dataset/voxcpm_pairs_short/m1nd3xpand3r_0000.wav"
        self.default_ref_text = None
        txt_file = self.default_ref_audio.replace(".wav", ".txt")
        if os.path.exists(txt_file):
            with open(txt_file) as f:
                self.default_ref_text = f.read().strip()
        print(f"✅ OmniVoice loaded: {model_path}")

    @modal.method()
    def generate(self, text: str, ref_audio: str = None, ref_text: str = None,
                 num_step: int = 32, guidance_scale: float = 3.5, speed: float = 1.0) -> dict:
        import tempfile, base64, time, soundfile as sf, numpy as np, os
        t0 = time.time()
        audio = self.model.generate(
            text=text, ref_audio=ref_audio or self.default_ref_audio,
            ref_text=ref_text or self.default_ref_text,
            num_step=num_step, guidance_scale=guidance_scale, speed=speed,
            denoise=True, postprocess_output=True,
        )
        if isinstance(audio, list):
            audio = audio[0]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        sf.write(out_path, audio, self.model.sampling_rate)
        wav_data, sr = sf.read(out_path, dtype="float32")
        with open(out_path, "rb") as f:
            wav_bytes = f.read()
        os.unlink(out_path)
        return {
            "audio_base64": base64.b64encode(wav_bytes).decode(),
            "audio_duration_sec": round(len(wav_data) / sr, 2),
            "generation_time_sec": round(time.time() - t0, 2),
        }


# ══════════════════════════════════════════════════════════════════════
# GPU Class 3: HiDream-O1 Image Gen (A10G)
# ══════════════════════════════════════════════════════════════════════
@app.cls(
    image=image_gen_image,
    gpu="A10G",
    volumes={"/cache": hf_cache},
    secrets=[hf_secret],
    scaledown_window=120,
    timeout=600,
    max_containers=1,
)
@modal.concurrent(max_inputs=2)
class HiDreamRunner:
    """HiDream-O1-Image — text-to-image up to 2048x2048."""

    @modal.enter()
    def load_model(self):
        import os, sys, torch, warnings
        from transformers import AutoProcessor

        # Suppress docstring warnings from HiDream's custom model code
        warnings.filterwarnings("ignore", message=".*is part of.*signature.*not documented.*")
        warnings.filterwarnings("ignore", message=".*torch_dtype is deprecated.*")

        # HiDream repo is pre-baked into the image at /workspace/HiDream-O1-Image
        repo_dir = "/workspace/HiDream-O1-Image"
        sys.path.insert(0, repo_dir)
        os.chdir(repo_dir)

        model_path = "HiDream-ai/HiDream-O1-Image-Dev"
        print(f"🎨 Loading {model_path}...")
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        tokenizer = self.processor.tokenizer
        for attr, val in [("boi_token", "<|boi_token|>"), ("bor_token", "<|bor_token|>"),
                          ("eor_token", "<|eor_token|>"), ("bot_token", "<|bot_token|>"),
                          ("tms_token", "<|tms_token|>")]:
            setattr(tokenizer, attr, val)
        self.repo_dir = repo_dir
        print("✅ HiDream loaded!")

    @modal.method()
    def generate(self, prompt: str, width: int = 2048, height: int = 2048, seed: int = 42) -> bytes:
        import io, sys, os
        os.chdir(self.repo_dir)
        sys.path.insert(0, self.repo_dir)
        from models.pipeline import generate_image, DEFAULT_TIMESTEPS
        image = generate_image(
            model=self.model, processor=self.processor, prompt=prompt,
            height=height, width=width, num_inference_steps=28, guidance_scale=0.0,
            shift=1.0, timesteps_list=DEFAULT_TIMESTEPS, scheduler_name="flash",
            seed=seed, noise_scale_start=7.5, noise_scale_end=7.5, noise_clip_std=2.5,
        )
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
# ASGI App
# ══════════════════════════════════════════════════════════════════════
def _create_web_app(brain_cls, tts_cls, img_cls):
    from fastapi import FastAPI, Body
    from fastapi.responses import Response, JSONResponse
    import base64, asyncio

    web = FastAPI(title="MindForge Omni v2 — Brain + Voice + Image", version="2.0")

    @web.get("/health")
    async def health():
        return {"status": "ok", "services": {
            "brain": {"model": "Qwen3.6-27B via OpenRouter", "role": "creative writing + prompt engineering"},
            "tts": {"model": "OmniVoice (checkpoint-1500)", "voice": "MindExpander clone",
                    "endpoint": "TheMindExpansionNetwork/omnivoice-mindexpander-voice"},
            "image": {"model": "HiDream-O1-Image-Dev", "max_resolution": "2048x2048"},
        }}

    # ── TTS ──
    @web.post("/v1/audio/speech")
    async def tts(text: str = Body(..., embed=True), voice: str = Body("mindexpander", embed=True),
                  num_step: int = Body(32, embed=True), guidance_scale: float = Body(3.5, embed=True)):
        runner = tts_cls()
        result = await runner.generate.remote.aio(text=text, num_step=num_step, guidance_scale=guidance_scale)
        return Response(content=base64.b64decode(result["audio_base64"]), media_type="audio/wav")

    # ── Image (with auto prompt enhancement) ──
    @web.post("/v1/images/generations")
    async def image_gen(prompt: str = Body(..., embed=True), size: str = Body("2048x2048", embed=True),
                       seed: int = Body(42, embed=True), enhance: bool = Body(True, embed=True)):
        final_prompt = prompt
        if enhance:
            brain = brain_cls()
            final_prompt = await brain.enhance_prompt.remote.aio(prompt)
        w, h = map(int, size.split("x"))
        runner = img_cls()
        img_bytes = await runner.generate.remote.aio(prompt=final_prompt, width=w, height=h, seed=seed)
        return JSONResponse(content={
            "data": [{"b64_json": base64.b64encode(img_bytes).decode()}],
            "original_prompt": prompt,
            "enhanced_prompt": final_prompt,
        })

    # ── Brain chat ──
    @web.post("/v1/brain/chat")
    async def brain_chat(messages: list = Body(..., embed=True), max_tokens: int = Body(2048, embed=True)):
        brain = brain_cls()
        result = await brain.chat.remote.aio(messages=messages, max_tokens=max_tokens)
        return JSONResponse(content={"response": result})

    # ── Full Omni Create — brain writes, voice narrates, image illustrates ──
    @web.post("/v1/omni/create")
    async def omni_create(
        request: str = Body("Write a poem about the cosmos and illustrate it", embed=True),
        style: str = Body("cinematic cyberpunk", embed=True),
    ):
        brain = brain_cls()
        creation = await brain.write_poem.remote.aio(theme=request, style=style)

        poem_text = creation.get("poem", request)
        image_prompt = creation.get("image_prompt",
            f"{style} illustration, {request}, highly detailed, dramatic lighting, 8k masterpiece")

        tts_runner = tts_cls()
        img_runner = img_cls()

        voice_task = tts_runner.generate.remote.aio(text=poem_text)
        image_task = img_runner.generate.remote.aio(prompt=image_prompt, width=2048, height=2048)

        voice_result, img_bytes = await asyncio.gather(voice_task, image_task)

        return JSONResponse(content={
            "title": creation.get("title", "Untitled"),
            "poem": poem_text,
            "image_prompt": image_prompt,
            "audio_base64": voice_result["audio_base64"],
            "audio_duration_sec": voice_result["audio_duration_sec"],
            "image_b64": base64.b64encode(img_bytes).decode(),
        })

    return web


@app.function(image=cpu_image, cpu=0.25, memory=512, scaledown_window=60, timeout=300)
@modal.asgi_app()
def api():
    brain_cls = modal.Cls.from_name(APP_NAME, "QwenBrain")
    tts_cls = modal.Cls.from_name(APP_NAME, "OmniVoiceRunner")
    img_cls = modal.Cls.from_name(APP_NAME, "HiDreamRunner")
    return _create_web_app(brain_cls, tts_cls, img_cls)
