"""Return a Mini-ImageNet image as PNG to stdout, given split and source_index."""
import sys
from datasets import load_dataset
from io import BytesIO

split = sys.argv[1]
index = int(sys.argv[2])

ds = load_dataset("timm/mini-imagenet", split=split)
image = ds[index]["image"].convert("RGB")
buf = BytesIO()
image.save(buf, format="PNG")
sys.stdout.buffer.write(buf.getvalue())
