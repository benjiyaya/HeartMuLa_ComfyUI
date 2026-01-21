# HeartMuLa_ComfyUI
ComfyUI Custom Node for HeartMuLa AI Music Generation and Transcript Text

**HeartMuLa** official GITHUB
https://github.com/HeartMuLa/heartlib


How To Use this In Basic: https://youtu.be/F9LFAeUbBIs


<img width="1418" height="595" alt="image" src="https://github.com/user-attachments/assets/44f4b065-bfe0-405d-8324-e10f5c60b320" />


<img width="1396" height="916" alt="image" src="https://github.com/user-attachments/assets/134a7776-6805-42a8-9e49-5852e8ee3ba9" />



------------------------------------------------------------
**Update:**

------------------------------------------------------------


2026-01-21 (2) 
- Lazy Load Optimization , now able to load with 12GB VRAM.  
- Path Configuration ,support custom model folder path in in the "extra_model_paths.yaml", Not limited by default ComfyUI/Models/  folder path only.

2026-01-21 
- MEMORY CLEANUP and Pipeline changed for BF16 - Optimized for 16GB. dtype I don't recommend under bf16 for this model, audio quality will degrade too much.


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

hf download HeartMuLa/HeartMuLaGen --local-dir ./HeartMuLa

hf download HeartMuLa/HeartMuLa-oss-3B --local-dir ./HeartMuLa/HeartMuLa-oss-3B

hf download HeartMuLa/HeartCodec-oss --local-dir ./HeartMuLa/HeartCodec-oss

hf download HeartMuLa/HeartTranscriptor-oss --local-dir ./HeartMuLa/HeartTranscriptor-oss


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






