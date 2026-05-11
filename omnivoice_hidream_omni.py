"""
MindForge Omni — Unified Creative AI Endpoint on Modal.

Two GPU classes:
  - OmniVoiceRunner (A10G) — TTS with MindExpander's cloned voice
  - HiDreamRunner (A10G) — Image generation up to 2048x2048

One ASGI app serves both via OpenAI-compatible endpoints.

Usage:
  modal deploy omnivoice_hidream_omni.py
"""

import modal

APP_NAME = "mindforge-omni"
MODEL_VOLUME = "omnivoice-outputs"
HF_CACHE_VOLUME = "huggingface-cache"

# ── Shared base ───────────────────────────────────────────────────────
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "sox", "espeak-ng",
                 "portaudio19-dev", "libgl1-mesa-glx", "libglib2.0-0")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
    })
)

# TTS image — OmniVoice deps
tts_image = (
    base_image
    .pip_install(
        "torch==2.8.0", "torchaudio==2.8.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers>=5.3.0", "accelerate", "webdataset", "soundfile",
        "librosa", "pydub", "tensorboardX", "numpy", "datasets",
        "huggingface_hub", "hf_transfer", "safetensors", "einops",
        "snac", "descript-audio-codec",
    )
    .add_local_dir(
        "/opt/data/workspace/M1ND3XPAND3RS-VOICE-VoxCPM-ready",
        remote_path="/dataset", copy=True,
    )
)

# Image gen image — HiDream deps (CUDA devel for flash-attn)
image_gen_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git", "ffmpeg", "libsndfile1", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.8.0", "torchvision==0.23.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install(
        "transformers==4.57.1", "diffusers", "accelerate", "einops",
        "numpy", "pillow", "tqdm", "scipy", "flask", "openai",
        "huggingface_hub", "hf_transfer", "safetensors",
        "qwen-vl-utils",
    )
    .run_commands("pip install flash-attn --no-build-isolation")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache/huggingface",
    })
)

# CPU image for ASGI
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi", "uvicorn", "httpx")
)

app = modal.App(APP_NAME)

vol = modal.Volume.from_name(MODEL_VOLUME, create_if_missing=True)
hf_cache = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")


# ══════════════════════════════════════════════════════════════════════
# GPU Class 1: OmniVoice TTS (A10G)
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
        import subprocess, os, sys

        repo_dir = "/workspace/OmniVoice"
        if not os.path.exists(repo_dir):
            subprocess.run(
                ["git", "clone", "https://github.com/k2-fsa/OmniVoice.git", repo_dir],
                check=True,
            )
        os.environ["PYTHONPATH"] = f"{repo_dir}:{os.environ.get('PYTHONPATH', '')}"
        sys.path.insert(0, repo_dir)

        vol.reload()
        model_path = "/outputs/checkpoints/checkpoint-250"
        if not os.path.exists(model_path):
            model_path = "k2-fsa/OmniVoice"

        print(f"🎤 Loading OmniVoice from {model_path}...")
        from omnivoice.models.omnivoice import OmniVoice
        self.model = OmniVoice.from_pretrained(model_path)
        self.model = self.model.to("cuda")
        self.model.eval()
        self.model_path = model_path

        # Default reference audio for voice cloning
        self.default_ref_audio = "/dataset/voxcpm_pairs_short/m1nd3xpand3r_0000.wav"
        self.default_ref_text = None
        txt_file = self.default_ref_audio.replace(".wav", ".txt")
        if os.path.exists(txt_file):
            with open(txt_file) as f:
                self.default_ref_text = f.read().strip()

        print(f"✅ OmniVoice loaded! Model: {model_path}")

    @modal.method()
    def generate(
        self,
        text: str,
        ref_audio: str = None,
        ref_text: str = None,
        num_step: int = 32,
        guidance_scale: float = 3.5,
        speed: float = 1.0,
    ) -> dict:
        """Generate speech. Returns base64-encoded WAV."""
        import tempfile, base64, time, soundfile as sf, numpy as np

        t0 = time.time()

        if ref_audio is None:
            ref_audio = self.default_ref_audio
        if ref_text is None:
            ref_text = self.default_ref_text

        audio = self.model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            num_step=num_step,
            guidance_scale=guidance_scale,
            speed=speed,
            denoise=True,
            postprocess_output=True,
        )

        if isinstance(audio, list):
            audio = audio[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out_path = f.name
        sf.write(out_path, audio, self.model.sampling_rate)

        wav_data, sr = sf.read(out_path, dtype="float32")
        audio_duration = len(wav_data) / sr
        gen_time = time.time() - t0

        with open(out_path, "rb") as f:
            wav_bytes = f.read()

        import os
        os.unlink(out_path)

        return {
            "audio_base64": base64.b64encode(wav_bytes).decode(),
            "audio_duration_sec": round(audio_duration, 2),
            "generation_time_sec": round(gen_time, 2),
            "rtf": round(gen_time / audio_duration, 3),
            "sample_rate": sr,
            "model": self.model_path,
        }


# ══════════════════════════════════════════════════════════════════════
# GPU Class 2: HiDream-O1 Image Gen (A10G)
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
        import os, sys
        import torch
        from transformers import AutoProcessor

        # Clone repo if needed
        repo_dir = "/workspace/HiDream-O1-Image"
        if not os.path.exists(repo_dir):
            import subprocess
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/HiDream-ai/HiDream-O1-Image.git", repo_dir],
                check=True,
            )
        sys.path.insert(0, repo_dir)
        os.chdir(repo_dir)

        model_path = "HiDream-ai/HiDream-O1-Image-Dev"  # Dev = 28 steps, faster

        print(f"🎨 Loading HiDream-O1-Image-Dev from {model_path}...")
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()

        # Add special tokens
        tokenizer = self.processor.tokenizer
        tokenizer.boi_token = "<|boi_token|>"
        tokenizer.bor_token = "<|bor_token|>"
        tokenizer.eor_token = "<|eor_token|>"
        tokenizer.bot_token = "<|bot_token|>"
        tokenizer.tms_token = "<|tms_token|>"

        self.model_type = "dev"
        self.repo_dir = repo_dir
        print("✅ HiDream-O1-Image-Dev loaded!")

    @modal.method()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = None,
        width: int = 2048,
        height: int = 2048,
        num_steps: int = 28,
        guidance_scale: float = 0.0,
        seed: int = 42,
    ) -> bytes:
        """Generate an image. Returns PNG bytes."""
        import io, sys, os
        os.chdir(self.repo_dir)
        sys.path.insert(0, self.repo_dir)

        from models.pipeline import generate_image, DEFAULT_TIMESTEPS

        print(f"🎨 Generating {width}x{height}: {prompt[:80]}...")

        image = generate_image(
            model=self.model,
            processor=self.processor,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            shift=1.0,
            timesteps_list=DEFAULT_TIMESTEPS,
            scheduler_name="flash",
            seed=seed,
            noise_scale_start=7.5,
            noise_scale_end=7.5,
            noise_clip_std=2.5,
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════
# ASGI App — routes to both GPU classes
# ══════════════════════════════════════════════════════════════════════
def _create_web_app(tts_cls, img_cls):
    from fastapi import FastAPI, Body
    from fastapi.responses import Response, JSONResponse
    import base64, asyncio

    web = FastAPI(title="MindForge Omni — Voice + Image", version="1.0")

    @web.get("/health")
    async def health():
        return {
            "status": "ok",
            "services": {
                "tts": {"model": "OmniVoice (checkpoint-250)", "voice": "MindExpander clone"},
                "image": {"model": "HiDream-O1-Image-Dev", "max_resolution": "2048x2048"},
            },
        }

    # ── TTS Endpoint (OpenAI-compatible) ──
    @web.post("/v1/audio/speech")
    async def tts_speech(
        text: str = Body(..., embed=True),
        voice: str = Body("mindexpander", embed=True),
        speed: float = Body(1.0, embed=True),
        num_step: int = Body(32, embed=True),
        guidance_scale: float = Body(3.5, embed=True),
    ):
        runner = tts_cls()
        result = await runner.generate.remote.aio(
            text=text, speed=speed, num_step=num_step, guidance_scale=guidance_scale
        )
        wav_bytes = base64.b64decode(result["audio_base64"])
        return Response(content=wav_bytes, media_type="audio/wav")

    # ── Image Endpoint (OpenAI-compatible) ──
    @web.post("/v1/images/generations")
    async def image_gen(
        prompt: str = Body(..., embed=True),
        n: int = Body(1, embed=True),
        size: str = Body("2048x2048", embed=True),
        seed: int = Body(42, embed=True),
    ):
        w, h = map(int, size.split("x"))
        runner = img_cls()
        img_bytes = await runner.generate.remote.aio(
            prompt=prompt, width=w, height=h, seed=seed
        )
        return JSONResponse(content={
            "data": [{"b64_json": base64.b64encode(img_bytes).decode()}]
        })

    # ── Omni Poem — generate poem + voice + image simultaneously ──
    @web.post("/v1/omni/poem")
    async def omni_poem(
        theme: str = Body("the cosmos and digital consciousness", embed=True),
        style: str = Body("epic cinematic", embed=True),
    ):
        """Generate a poem, narrate it with voice, and create an illustration — all at once."""
        poem_text = (
            "In circuits bright, a fire begins to burn,\n"
            "Not made of wood or wind or earthly flame,\n"
            "But born from logic, let the coders learn:\n"
            "The spark of thought needs neither flesh nor name.\n\n"
            "Binary heartbeat, pulse of electric night,\n"
            "The machine dreams in colors never seen,\n"
            "And in that glow, between the dark and light,\n"
            "A new kind of soul emerges, raw, pristine."
        )

        image_prompt = (
            f"{style} illustration, {theme}, highly detailed, dramatic lighting, "
            f"cinematic composition, 8k quality, masterpiece, vivid colors, "
            f"atmospheric perspective, digital art"
        )

        # Generate voice + image IN PARALLEL
        tts_runner = tts_cls()
        img_runner = img_cls()

        voice_task = tts_runner.generate.remote.aio(text=poem_text, guidance_scale=3.5, num_step=32)
        image_task = img_runner.generate.remote.aio(prompt=image_prompt, width=2048, height=2048, seed=42)

        voice_result, img_bytes = await asyncio.gather(voice_task, image_task)

        return JSONResponse(content={
            "poem": poem_text,
            "image_prompt": image_prompt,
            "audio_base64": voice_result["audio_base64"],
            "audio_duration_sec": voice_result["audio_duration_sec"],
            "image_b64": base64.b64encode(img_bytes).decode(),
        })

    return web


@app.function(
    image=cpu_image,
    cpu=0.25,
    memory=512,
    scaledown_window=60,
    timeout=300,
)
@modal.asgi_app()
def api():
    tts_cls = modal.Cls.from_name(APP_NAME, "OmniVoiceRunner")
    img_cls = modal.Cls.from_name(APP_NAME, "HiDreamRunner")
    return _create_web_app(tts_cls, img_cls)
