"""
Subsystem B: CNN Flood Detection from Satellite Images
Uses Leslie's trained PyTorch U-Net model
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path


class ConvBlock(nn.Module):
    """Double convolution block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block - matches Leslie's pattern."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Leslie's pattern: upsample FROM in_ch TO out_ch, then concat makes in_ch total
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        # After concat with skip: out_ch + out_ch = in_ch for conv input
        self.conv = ConvBlock(out_ch * 2, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture matching Leslie's model."""
    
    def __init__(self, in_channels=9, out_channels=1, base_channels=32):
        super().__init__()
        
        # Encoder - goes UP: 32 → 64 → 128 → 256
        self.enc1 = ConvBlock(in_channels, base_channels)           # 32
        self.enc2 = ConvBlock(base_channels, base_channels * 2)     # 64
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4) # 128
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8) # 256
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)  # 512
        
        # Decoder - goes DOWN: 256 → 128 → 64 → 32 (REVERSED to match Leslie)
        self.dec4 = UpBlock(base_channels * 16, base_channels * 8)  # 512→256
        self.dec3 = UpBlock(base_channels * 8, base_channels * 4)   # 256→128
        self.dec2 = UpBlock(base_channels * 4, base_channels * 2)   # 128→64
        self.dec1 = UpBlock(base_channels * 2, base_channels)       # 64→32
        
        # Output
        self.out = nn.Conv2d(base_channels, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder - REVERSED order to match Leslie
        dec4 = self.dec4(bottleneck, enc4)
        dec3 = self.dec3(dec4, enc3)
        dec2 = self.dec2(dec3, enc2)
        dec1 = self.dec1(dec2, enc1)
        
        return self.sigmoid(self.out(dec1))


class FloodDetector:
    """Wrapper for Leslie's flood detection model."""
    
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.model_path = Path(model_path)
        self.img_size = 512
        self.threshold = 0.45
        
        if self.model_path.exists():
            self.load_model()
    
    def load_model(self):
        """Load the trained PyTorch model."""
        try:
            self.model = UNet(in_channels=9, out_channels=1, base_channels=32)
            
            # Load weights
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Load with strict=False to handle any minor mismatches
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ Leslie's CNN model loaded successfully!")
            if missing_keys:
                print(f"  Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"  Unexpected keys: {len(unexpected_keys)}")
            
        except Exception as e:
            print(f"✗ Failed to load CNN model: {e}")
            print("⚠️  Using MOCK mode - real model architecture mismatch")
            print("    Will generate random predictions for demo purposes")
            self.model = "MOCK"
    
    def preprocess_image(self, image_bytes):
        """Preprocess uploaded image for model input."""
        # Load image
        img = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 512x512
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # RGB to 9 channels (simulate multi-spectral by repeating/augmenting)
        # In production, this would be actual Sentinel-2 bands
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Create 9 pseudo-channels from RGB
        channels = [
            r,                          # Red
            g,                          # Green
            b,                          # Blue
            (r + g + b) / 3,           # Panchromatic
            np.clip(g - r, 0, 1),      # NDVI-like
            np.clip(r + g, 0, 1),      # Brightness
            np.clip(b * 1.2, 0, 1),    # Enhanced blue (water)
            np.clip(g * 1.1, 0, 1),    # Enhanced green
            np.clip((r + b) / 2, 0, 1) # RB combination
        ]
        
        # Stack into 9-channel tensor
        img_tensor = np.stack(channels, axis=0)
        img_tensor = torch.from_numpy(img_tensor).unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device)
    
    def predict(self, image_bytes):
        """Predict flood probability from uploaded image."""
        if self.model is None:
            return None, "Model not loaded"
        
        # MOCK MODE - generate demo predictions
        if self.model == "MOCK":
            try:
                # Just for UI demo - random but realistic predictions
                import random
                flood_percentage = random.uniform(15, 45)
                avg_confidence = random.uniform(60, 85)
                max_confidence = random.uniform(85, 95)
                
                # Determine risk level
                if flood_percentage < 10:
                    risk_level = "Green"
                elif flood_percentage < 30:
                    risk_level = "Yellow"
                elif flood_percentage < 60:
                    risk_level = "Orange"
                else:
                    risk_level = "Red"
                
                risk_conf = avg_confidence / 100
                
                # Create dummy probability map
                dummy_map = [[random.random() * 0.6 for _ in range(32)] for _ in range(32)]
                dummy_binary = [[1 if v > 0.45 else 0 for v in row] for row in dummy_map]
                
                result = {
                    'flood_percentage': flood_percentage,
                    'avg_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'risk_level': risk_level,
                    'risk_confidence': risk_conf,
                    'probability_map': dummy_map,
                    'binary_mask': dummy_binary,
                    'explanation': [
                        "⚠️  DEMO MODE - Using placeholder predictions",
                        f"Simulated flood detection: {flood_percentage:.1f}% of area",
                        f"Simulated confidence: {avg_confidence:.1f}%",
                        "Real predictions require Leslie's model architecture",
                        "Contact Leslie for exact UNet parameters"
                    ]
                }
                
                return result, None
                
            except Exception as e:
                return None, str(e)
        
        # REAL MODEL MODE (when we get the right architecture)
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image_bytes)
            
            # Predict
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # Convert to numpy
            pred_mask = output.squeeze().cpu().numpy()
            
            # Calculate metrics
            flood_percentage = float((pred_mask > self.threshold).mean() * 100)
            avg_confidence = float(pred_mask.mean() * 100)
            max_confidence = float(pred_mask.max() * 100)
            
            # Binary mask
            binary_mask = (pred_mask > self.threshold).astype(np.uint8)
            
            # Determine risk level
            if flood_percentage < 10:
                risk_level = "Green"
                risk_conf = avg_confidence / 100
            elif flood_percentage < 30:
                risk_level = "Yellow"
                risk_conf = avg_confidence / 100
            elif flood_percentage < 60:
                risk_level = "Orange"
                risk_conf = avg_confidence / 100
            else:
                risk_level = "Red"
                risk_conf = avg_confidence / 100
            
            result = {
                'flood_percentage': flood_percentage,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'risk_level': risk_level,
                'risk_confidence': risk_conf,
                'probability_map': pred_mask.tolist(),
                'binary_mask': binary_mask.tolist(),
                'explanation': [
                    f"Satellite image analysis complete (512×512 pixels)",
                    f"Flood detected in {flood_percentage:.1f}% of area",
                    f"Average flood confidence: {avg_confidence:.1f}%",
                    f"Peak flood confidence: {max_confidence:.1f}%",
                    f"Risk threshold: {self.threshold * 100:.0f}%"
                ]
            }
            
            return result, None
            
        except Exception as e:
            return None, str(e)
