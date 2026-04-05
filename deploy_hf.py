"""Upload updated app.py + new V2 models to Hugging Face Spaces."""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import HfApi

TOKEN = os.getenv("HF_TOKEN")  # Security: Using environment variable
REPO_ID = "Govindapawar07/stress-detection"
BASE = r"c:\Users\atulk\OneDrive\Desktop\Stress Detection"

api = HfApi(token=TOKEN)

# Base files that must upload
files_to_upload = [
    ("app.py",               "app.py"),       
    ("efficientnet_b2.pth",  "efficientnet_b2.pth"),
]

# Check if V2 models exist, if so upload them!
FACE_V2 = os.path.join(BASE, "models", "face_model_v2.pth")
VOICE_V2 = os.path.join(BASE, "models", "voice_model_v2.pth")

if os.path.exists(FACE_V2):
    files_to_upload.append((FACE_V2, "models/face_model_v2.pth"))
    
if os.path.exists(VOICE_V2):
    files_to_upload.append((VOICE_V2, "models/voice_model_v2.pth"))

print("🚀 Starting Uploads to Hugging Face...")

for local_path, remote_name in files_to_upload:
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    print(f"📤 Uploading {remote_name} ({size_mb:.1f} MB)...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_name,
        repo_id=REPO_ID,
        repo_type="space",
        token=TOKEN,
    )
    print(f"   ✅ Done: {remote_name}")

print(f"\n🎉 All done! Your upgraded Space is live at:")
print(f"👉 https://huggingface.co/spaces/{REPO_ID}")
