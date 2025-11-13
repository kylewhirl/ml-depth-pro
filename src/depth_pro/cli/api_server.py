import io
import logging
import tempfile
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from depth_pro import create_model_and_transforms, load_rgb
from depth_pro.cli.run import get_torch_device  # reuse your helper

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="DepthPro Grayscale Depth API")

# ---- Model init (happens once, when the server starts) ----------------------
LOGGER.info("Loading DepthPro model...")
device = get_torch_device()
model, transform = create_model_and_transforms(
    device=device,
    precision=torch.half,
)
model.eval()
LOGGER.info("DepthPro model loaded on device %s", device)


def _predict_grayscale_depth(image_path: Path) -> bytes:
    """
    Run DepthPro on a single image and return an 8-bit grayscale PNG (bytes).
    Nearer = brighter. This mirrors your --grayscale branch from run.py.
    """
    # Load image using the same helper as the CLI
    image, _, f_px = load_rgb(image_path)

    with torch.no_grad():
        prediction = model.infer(transform(image), f_px=f_px)

    depth = prediction["depth"].detach().cpu().numpy().squeeze()

    # Inverse depth normalization (same logic as in run.py)
    inverse_depth = 1.0 / depth
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )
    inverse_depth_normalized = np.clip(inverse_depth_normalized, 0.0, 1.0)

    gray = (inverse_depth_normalized * 255).astype(np.uint8)
    img = PIL.Image.fromarray(gray)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@app.post("/depth-map", summary="Generate grayscale depth PNG from an uploaded image")
async def depth_map(file: UploadFile = File(...)):
    """
    Upload an image and get back a grayscale depth map as image/png.
    """
    try:
        contents = await file.read()
        if not contents:
            raise ValueError("Uploaded file is empty")

        # Write to a temporary file so we can reuse load_rgb(image_path)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / (file.filename or "input.png")
            tmp_path.write_bytes(contents)

            png_bytes = _predict_grayscale_depth(tmp_path)

    except Exception as e:
        LOGGER.exception("DepthPro inference failed")
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=png_bytes, media_type="image/png")
