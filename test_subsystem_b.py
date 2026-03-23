from subsystem_b import FloodDetector
from PIL import Image
import io

# Create a test image (blue water-like image)
img = Image.new('RGB', (512, 512), color=(50, 100, 200))
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Test prediction
detector = FloodDetector('models/best_flood_early_warning_unet.pt')
result, error = detector.predict(img_bytes.getvalue())

if error:
    print(f"ERROR: {error}")
else:
    print(f"✓ Flood Coverage: {result['flood_percentage']:.1f}%")
    print(f"✓ Avg Confidence: {result['avg_confidence']:.1f}%")
    print(f"✓ Risk Level: {result['risk_level']}")
    print(f"✓ Explanation: {result['explanation'][0]}")
