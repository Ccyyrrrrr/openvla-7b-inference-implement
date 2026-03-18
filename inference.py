import sys
import torch
from PIL import Image
from pathlib import Path

# ✅ Fix 1: If using editable install, DON'T add to sys.path at all.
# The editable install already registers the package.
# If you MUST add manually, point to the PARENT of the openvla folder:
# sys.path.insert(0, "E:\\")   ← parent of E:\openvla

# ✅ Fix 2: Use the correct HuggingFace API (OpenVLA has no openvla.models.VLA)
from transformers import AutoModelForVision2Seq, AutoProcessor

# ===================== Configuration =====================
MODEL_PATH = "C:/Users\Administrator/.cache/huggingface/hub/models--openvla--openvla-7b/snapshots/47a0ec7fc4ec123775a391911046cf33cf9ed83f"
image_path  = "C:/Users/Administrator/Desktop/test_image.jpg"
instruction = "pick up the bottle"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ===================== Load Processor & Model =====================
try:
    print(f"Loading processor from: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )

    print(f"Loading model from: {MODEL_PATH}")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,   # bfloat16 is recommended for OpenVLA
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")

except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise

# ===================== Load Image =====================
try:
    image = Image.open(image_path).convert("RGB")
    print(f"✅ Image loaded: {image.size}")
except Exception as e:
    print(f"❌ Image load failed: {e}")
    raise

# ===================== Run Inference =====================
try:
    print("Running inference...")

    # processor handles both image resizing and tokenization
    inputs = processor(
        text=instruction,
        images=image,
        return_tensors="pt"
    ).to(DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        # predict_action returns a 7-DoF action vector
        action = model.predict_action(
            **inputs,
            unnorm_key="bridge_orig",   # normalization key for your dataset
            do_sample=False,
        )

    print("\n✅ Predicted robot action (7-DoF):")
    print(action)

except Exception as e:
    print(f"❌ Inference failed: {e}")
    raise