---
name: ai-media
description: AI image and video generation on a local GPU (ComfyUI or diffusers); not yet tried here
status: experimental
---
EXPERIMENTAL: this has not been run in this setup. Treat everything below as a plan to verify, and tell the user it is untested.
1. Check the hardware first: `nvidia-smi` for VRAM, free disk for models, and whether torch with CUDA is installed. A 16 GB card can run SDXL-class image models; video models need more patience or more VRAM.
2. Install ComfyUI or diffusers in a fresh venv on the GPU machine, not on a Raspberry Pi.
3. Start with one image at low resolution to prove the pipeline end to end, then raise quality. Record the model, prompt, seed and settings next to every output so results can be repeated.
4. For video, generate short clips (a few seconds), then assemble with the video-reel skill's ffmpeg steps.
5. Check licences of the model weights before anything commercial.
When this has actually been run once successfully, change `status` to tested in this file and write down the exact versions and commands that worked.
