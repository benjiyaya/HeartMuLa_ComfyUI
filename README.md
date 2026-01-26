# Project Overview

This repository serves as the official implementation of HeartMuLA (Multi-Attention Latent Alignment) specifically tailored for the ComfyUI ecosystem. It is designed to facilitate historical image synthesis and texture generation by leveraging advanced latent space alignment techniques. By integrating this custom node, users can access the capabilities of the HeartMuLA AI model directly within their node-based workflows, enabling the creation of distinct artistic styles with high precision.

# HeartMuLa_ComfyUI

HeartMuLA_ComfyUI is a custom node designed to significantly expand the feature set of ComfyUI, the popular node-based diffusion interface. This custom integration focuses on implementing Multi-Attention Latent Alignment, enabling users to achieve precise control over image generation and style transfer. By adding HeartMuLA to your ComfyUI workflow, you gain access to advanced alignment techniques that manipulate the latent space with high accuracy, allowing for the creation of distinct artistic styles and complex visual structures that standard workflows struggle to replicate.

At the core of this implementation is the sophisticated HeartMuLA AI model, which drives the alignment process to refine image outputs with exceptional fidelity. Rephrasing the essential setup instructions, users can easily deploy this AI model by cloning the repository directly into the custom_nodes folder within their ComfyUI directory. This seamless integration eliminates the need for complex manual configurations, allowing you to simply load the node and start utilizing the feature immediately. Whether you are looking to experiment with historical art styles or specific texture synthesis, this custom node bridges the gap between state-of-the-art research and practical, usable tools within the ComfyUI ecosystem.

**HeartMuLa** official GITHUB
https://github.com/HeartMuLa/heartlib

How To Use this In Basic: https://youtu.be/F9LFAeUbBIs


Features - Music Generation

<img width="1826" height="1016" alt="Screenshot 2026-01-22 214655" src="https://github.com/user-attachments/assets/cbdce761-8952-4ce1-b667-17e8e59db96c" />

Lyrics Transcript

<img width="1418" height="595" alt="image" src="https://github.com/user-attachments/assets/44f4b065-bfe0-405d-8324-e10f5c60b320" />


------------------------------------------------------------
**Message**
------------------------------------------------------------

Let's make this project the true open source, anyone who is interested in making improvements feel free to let us know in Discussions : the https://github.com/benjiyaya/HeartMuLa_ComfyUI/discussions

We are not providing service here, we are group of hobbiests, developers who want to make something here.  So don't take it for granted.

------------------------------------------------------------
**Update:**

------------------------------------------------------------
2026-01-23 :
Support new model from HeartMuLa

HeartMuLa-RL-oss-3B-20260123

https://huggingface.co/HeartMuLa/HeartMuLa-RL-oss-3B-20260123

HeartCodec-oss-20260123

https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123


2026-01-22 (2) :
thank you [@zboyles](https://github.com/zboyles) for making this custom node support Apple M-series!

2026-01-22 :
Feature: 4-bit Quantization (FP4/NF4) with Native Blackwell Detection
Thank you <a href="https://github.com/IuraHD">IuraHD</a> update!

- some information about FP4 Computing here : [FP4 Compute](https://github.com/benjiyaya/HeartMuLa_ComfyUI/blob/main/README_FP4.md)

<img width="1144" height="639" alt="image" src="https://github.com/user-attachments/assets/0bc79126-c37b-4ad0-8e70-947172e49d5d" />



2026-01-21 (3)
- Integrates native progress bars, making it easy to implement real-time progress tracking in the user interface.
- Precise Temperature: Refined temperature step to 0.01 for more granular control over generation.
- Intuitive Audio Length: Renamed parameter to max_audio_length_seconds (Default: 240s) for better usability.
- Keep Model Loaded(Memory Settings):
True: Keep model in VRAM for instant subsequent generations.
False: Unload model after each task to free up memory.
- Offload Mode:
Auto: Standard memory release for balanced performance.
Aggressive: Full VRAM wipe + Garbage Collection. 

2026-01-21 (2) 
- Lazy Load Optimization , now able to load with 12GB VRAM.  
- Path Configuration ,support custom model folder path in in the "extra_model_paths.yaml", Not limited by default ComfyUI/Models/  folder path only.

2026-01-21 
- MEMORY CLEANUP and Pipeline changed for BF16 - Optimized for 16GB. dtype I don't recommand under bf16 for this model, audio quality will degrade too much.


------------------------------------------------------------

# Installation

------------------------------------------------------------

**Step 1**

Go to ComfyUI\custom_nodes
Command prompt:

git clone https://github.com/benjiyaya/HeartMuLa_ComfyUI

**Step 2**

cd /HeartMuLa_ComfyUI

**Step 3**

pip install -r requirements.txt

If no module name error pop up.
some libraries might need to install Individually (For Windows users you need use Command Prompt as Administrator)

do this :

pip install soundfile

pip install torchtune

pip install torchao

For Windows User, Download a "full-shared" build of FFmpeg. Ensure you extract it and add the bin folder (containing the .dll files) to your system Path. 
https://github.com/GyanD/codexffmpeg/releases/tag/8.0.1

------------------------------------------------------------

# For File structure

------------------------------------------------------------

<img width="1179" height="345" alt="image" src="https://github.com/user-attachments/assets/5087e10e-9815-48ff-bbb4-3a21dc1e54d1" />


------------------------------------------------------------

# Download model files

------------------------------------------------------------
Go to ComfyUI/models 

Use HuggingFace Cli download model weights.

type :

1 - HeartMuLaGen

hf download HeartMuLa/HeartMuLaGen --local-dir ./HeartMuLa



2 - HeartMuLaGen base model

hf download HeartMuLa/HeartMuLa-oss-3B --local-dir ./HeartMuLa/HeartMuLa-oss-3B

or 

hf download HeartMuLa/HeartMuLa-RL-oss-3B-20260123 --local-dir ./HeartMuLa/HeartMuLa-RL-oss-3B-20260123



3 - HeartCodec model

hf download HeartMuLa/HeartCodec-oss --local-dir ./HeartMuLa/HeartCodec-oss

or

hf download HeartMuLa/HeartCodec-oss-20260123 --local-dir ./HeartMuLa/HeartCodec-oss-20260123



4 - HeartTranscriptor

hf download HeartMuLa/HeartTranscriptor-oss --local-dir ./HeartMuLa/HeartTranscriptor-oss




***If you download oss-20260123 for HeartMuLa-3B,  you must need to use HeartCodec-oss-20260123


------------------------------------------------------------

# For Model File structure

------------------------------------------------------------


<img width="1391" height="320" alt="image" src="https://github.com/user-attachments/assets/3b48ff70-2a4f-4f8d-aed2-d0fbc76bb31f" />



------------------------------------------------------------


Model Sources
------------------------------------------------------------

Github Repo: https://github.com/HeartMuLa/heartlib

Paper: https://arxiv.org/abs/2601.10547

Demo: https://heartmula.github.io/

HeartMuLa-oss-3B: https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B

HeartCodec-oss: https://huggingface.co/HeartMuLa/HeartCodec-oss

HeartTranscriptor-oss: https://huggingface.co/HeartMuLa/HeartTranscriptor-oss







Credits
------------------------------------------------------------
HeartMuLa: https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B






