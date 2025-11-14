import argparse
import random
import csv
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

def make_dataset(out_dir, img_size=128, num_per_combo=20):
    """Enhanced dataset with stronger color signals."""
    out_dir = Path(out_dir)
    images_dir = out_dir / "multimodal_dataset" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # VIBRANT colors (higher saturation)
    colors = {
        "red": (255, 0, 0),      # Pure red
        "blue": (0, 0, 255),     # Pure blue
        "green": (0, 255, 0),    # Pure green
        "yellow": (255, 255, 0), # Pure yellow
    }
    
    shapes = ["circle", "square", "triangle"]
    annotations = []
    idx = 0
    
    random.seed(42)
    np.random.seed(42)
    
    for cname, crgb in colors.items():
        for shape in shapes:
            for _ in range(num_per_combo):
                # White background
                img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                
                # Larger shapes for better visibility
                cx, cy = img_size // 2, img_size // 2
                w = int(img_size * 0.5)  # 50% of image size
                bbox = (cx - w//2, cy - w//2, cx + w//2, cy + w//2)
                
                if shape == "circle":
                    draw.ellipse(bbox, fill=crgb, outline=crgb)
                elif shape == "square":
                    draw.rectangle(bbox, fill=crgb, outline=crgb)
                else:  # triangle
                    xA, yA = cx, cy - w//2
                    xB, yB = cx - w//2, cy + w//2
                    xC, yC = cx + w//2, cy + w//2
                    draw.polygon([(xA,yA),(xB,yB),(xC,yC)], fill=crgb, outline=crgb)
                
                fname = f"img_{idx:04d}.png"
                img.save(images_dir / fname)
                annotations.append([f"images/{fname}", f"{cname} {shape}", cname])
                idx += 1
    
    ann_file = out_dir / "multimodal_dataset" / "annotations.csv"
    with open(ann_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "caption", "color"])
        w.writerows(annotations)
    
    print(f"[✓] Dataset: {idx} images with pure colors")
    return out_dir / "multimodal_dataset", images_dir, ann_file

class ShapesDataset(Dataset):
    def __init__(self, root_dir, annotations_csv, transform=None):
        import pandas as pd
        
        self.root = Path(root_dir)
        df = pd.read_csv(annotations_csv)
        self.items = df.to_dict(orient="records")
        
        # Normalize to [-1, 1]
        self.transform = transform if transform else T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Build vocab
        self.vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2
        for item in self.items:
            for w in item["caption"].split():
                if w not in self.vocab:
                    self.vocab[w] = idx
                    idx += 1
        
        # Color vocabulary
        self.color_vocab = {"red": 0, "blue": 1, "green": 2, "yellow": 3}
        self.max_len = 4
        
        print(f"[✓] Vocab: {len(self.vocab)} tokens, 4 colors")
    
    def caption_to_tensor(self, caption):
        words = caption.split()
        ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in words]
        while len(ids) < self.max_len:
            ids.append(self.vocab["<pad>"])
        return torch.tensor(ids[:self.max_len], dtype=torch.long)
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        row = self.items[idx]
        img = Image.open(self.root / row["filename"]).convert("RGB")
        img = self.transform(img)
        cap = self.caption_to_tensor(row["caption"])
        
        # Color label for conditioning
        color_idx = self.color_vocab.get(row["color"], 0)
        color_label = torch.tensor(color_idx, dtype=torch.long)
        
        return img, cap, color_label, row["caption"]

class ColorAwareTextEncoder(nn.Module):
    def __init__(self, vocab_size, num_colors=4, embed_dim=128, hidden_dim=256, latent_dim=512):
        super().__init__()
        
        # Text embedding
        self.text_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Color embedding (separate pathway)
        self.color_embed = nn.Embedding(num_colors, 64)
        
        # LSTM for text
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=2, batch_first=True, bidirectional=True
        )
        
        # Fusion layer: combines text + color
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 64, latent_dim),
            nn.LayerNorm(latent_dim),  # Better than BatchNorm for stability
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
    
    def forward(self, text_ids, color_label):
        # Text encoding
        text_emb = self.text_embed(text_ids)  # [B, L, 128]
        _, (h_n, _) = self.lstm(text_emb)
        h_text = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 512]
        
        # Color encoding
        h_color = self.color_embed(color_label)  # [B, 64]
        
        # Fuse text + color
        h_combined = torch.cat([h_text, h_color], dim=1)  # [B, 576]
        z = self.fusion(h_combined)  # [B, 512]
        
        return z

class ColorPreservingGenerator(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.ReLU()
        )
        
        # Using InstanceNorm2d instead of BatchNorm2d
        # InstanceNorm: normalizes per-sample → preserves colors
        # BatchNorm: normalizes per-batch → washes out colors
        self.deconv = nn.Sequential(
            # 4 → 8
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),  # ← KEY FIX
            nn.ReLU(),
            
            # 8 → 16
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),  # ← KEY FIX
            nn.ReLU(),
            
            # 16 → 32
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),   # ← KEY FIX
            nn.ReLU(),
            
            # 32 → 64
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.InstanceNorm2d(32),   # ← KEY FIX
            nn.ReLU(),
            
            # 64 → 128
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.InstanceNorm2d(16),   # ← KEY FIX
            nn.ReLU(),
            
            # Final layer
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)
        x = self.deconv(x)
        return x

class ColorFocusedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # 1. Pixel L1 loss
        l1_loss = self.l1(pred, target)
        
        # 2. Channel mean matching (color distribution)
        pred_means = pred.mean(dim=[2, 3])  # [B, 3]
        target_means = target.mean(dim=[2, 3])
        mean_loss = self.mse(pred_means, target_means)
        
        # 3. Channel std matching (color intensity)
        pred_stds = pred.std(dim=[2, 3])
        target_stds = target.std(dim=[2, 3])
        std_loss = self.mse(pred_stds, target_stds)
        
        # 4. Dominant color penalty
        # Find max channel in target and ensure it's max in pred
        target_max_channel = target.mean(dim=[2, 3]).argmax(dim=1)  # [B]
        pred_channel_vals = pred.mean(dim=[2, 3])  # [B, 3]
        
        # Get predicted values for the dominant channel
        dominant_vals = pred_channel_vals[torch.arange(pred.size(0)), target_max_channel]
        
        # Ensure dominant channel has highest value
        other_vals = pred_channel_vals.sum(dim=1) - dominant_vals
        dominance_loss = F.relu(other_vals - dominant_vals * 2).mean()
        
        # Weighted combination (emphasize color)
        total_loss = (
            0.5 * l1_loss +           # Pixel accuracy
            2.0 * mean_loss +         # Color distribution ← INCREASED
            1.0 * std_loss +          # Color intensity
            1.5 * dominance_loss      # Dominant color ← NEW
        )
        
        return total_loss

class Evaluator:
    def __init__(self):
        self.color_map = {
            "red": torch.tensor([1.0, -1.0, -1.0]),
            "blue": torch.tensor([-1.0, -1.0, 1.0]),
            "green": torch.tensor([-1.0, 1.0, -1.0]),
            "yellow": torch.tensor([1.0, 1.0, -1.0])
        }
    
    def color_accuracy(self, images, color_labels):
        batch_size = images.size(0)
        correct = 0
        
        for i in range(batch_size):
            img = images[i]  # [3, H, W]
            color_name = color_labels[i]
            
            # Get mean RGB values
            r = img[0].mean().item()
            g = img[1].mean().item()
            b = img[2].mean().item()
            
            # Determine dominant channel
            if color_name == "red" and r > g and r > b:
                correct += 1
            elif color_name == "blue" and b > r and b > g:
                correct += 1
            elif color_name == "green" and g > r and g > b:
                correct += 1
            elif color_name == "yellow" and r > 0 and g > 0 and b < 0:
                correct += 1
        
        return correct / batch_size
    
    def clip_score_simple(self, images, captions):
        # Simplified: check if image colors match caption colors
        scores = []
        
        for img, cap in zip(images, captions):
            # Extract color from caption
            cap_lower = cap.lower()
            
            # Get dominant color in image
            r = img[0].mean().item()
            g = img[1].mean().item()
            b = img[2].mean().item()
            
            score = 0.0
            if "red" in cap_lower and r > max(g, b):
                score += 1.0
            elif "blue" in cap_lower and b > max(r, g):
                score += 1.0
            elif "green" in cap_lower and g > max(r, b):
                score += 1.0
            elif "yellow" in cap_lower and r > 0 and g > 0:
                score += 1.0
            
            scores.append(score)
        
        return np.mean(scores)

def train(args):
    print("[*] Initializing enhanced training...")
    root = Path(args.out_dir)
    
    if not (root / "multimodal_dataset").exists():
        make_dataset(root, num_per_combo=args.num_per_combo)
    
    ds_root = root / "multimodal_dataset"
    ann = ds_root / "annotations.csv"
    ds = ShapesDataset(ds_root, ann)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Device: {device}")
    
    # Initialize models
    enc = ColorAwareTextEncoder(len(ds.vocab), num_colors=4).to(device)
    gen = ColorPreservingGenerator().to(device)
    
    # Optimizer
    params = list(enc.parameters()) + list(gen.parameters())
    optimizer = optim.AdamW(params, lr=2e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Loss and evaluator
    criterion = ColorFocusedLoss()
    evaluator = Evaluator()
    
    print("[*] Training with color-focused loss...")
    best_loss = float('inf')
    best_color_acc = 0.0
    
    for epoch in range(args.epochs):
        enc.train()
        gen.train()
        
        total_loss = 0
        all_images = []
        all_color_labels = []
        all_captions = []
        
        for imgs, caps, color_labels, captions in dl:
            imgs = imgs.to(device)
            caps = caps.to(device)
            color_labels = color_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with color conditioning
            z = enc(caps, color_labels)
            pred = gen(z)
            
            loss = criterion(pred, imgs)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Collect for evaluation
            all_images.append(pred.detach().cpu())
            all_color_labels.extend(captions)
            all_captions.extend(captions)
        
        scheduler.step()
        avg_loss = total_loss / len(dl)
        
        # Evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            enc.eval()
            gen.eval()
            
            with torch.no_grad():
                sample_imgs = torch.cat(all_images[:4])
                sample_colors = all_color_labels[:4 * args.batch_size]
                sample_caps = all_captions[:4 * args.batch_size]
                
                color_acc = evaluator.color_accuracy(
                    sample_imgs, 
                    [c.split()[0] for c in sample_colors]
                )
                clip_score = evaluator.clip_score_simple(sample_imgs, sample_caps)
            
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"ColorAcc: {color_acc:.2%} | CLIP: {clip_score:.3f}")
            
            if color_acc > best_color_acc:
                best_color_acc = color_acc
                torch.save({
                    "encoder": enc.state_dict(),
                    "generator": gen.state_dict(),
                    "vocab": ds.vocab,
                    "color_vocab": ds.color_vocab,
                    "epoch": epoch,
                    "color_acc": color_acc
                }, root / "models_best.pth")
        else:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    torch.save({
        "encoder": enc.state_dict(),
        "generator": gen.state_dict(),
        "vocab": ds.vocab,
        "color_vocab": ds.color_vocab,
        "epoch": args.epochs
    }, root / "models_final.pth")
    
    print(f"\n[✓] Training complete!")
    print(f"    Best Color Accuracy: {best_color_acc:.2%}")
    print(f"    Best Loss: {best_loss:.4f}")

def generate(args):
    root = Path(args.out_dir)
    model_path = root / "models_best.pth"
    
    if not model_path.exists():
        model_path = root / "models_final.pth"
    
    if not model_path.exists():
        print("[!] No model found. Train first with --train")
        return
    
    state = torch.load(model_path, map_location="cpu")
    vocab = state["vocab"]
    color_vocab = state["color_vocab"]
    
    enc = ColorAwareTextEncoder(len(vocab), num_colors=4)
    gen = ColorPreservingGenerator()
    
    enc.load_state_dict(state["encoder"])
    gen.load_state_dict(state["generator"])
    
    enc.eval()
    gen.eval()
    
    results = root / "results"
    results.mkdir(exist_ok=True)
    
    samples = args.samples.split(",") if args.samples else [
        "red circle", "blue square", "green triangle", "yellow circle",
        "red square", "blue triangle", "green circle", "yellow square"
    ]
    
    print(f"[*] Generating {len(samples)} samples...")
    
    with torch.no_grad():
        for i, cap in enumerate(samples):
            words = cap.strip().split()
            color_name = words[0] if words else "red"
            
            # Tokenize
            ids = [vocab.get(w, vocab.get("<unk>", 1)) for w in words]
            while len(ids) < 4:
                ids.append(vocab.get("<pad>", 0))
            
            cap_tensor = torch.tensor([ids[:4]], dtype=torch.long)
            color_idx = color_vocab.get(color_name, 0)
            color_tensor = torch.tensor([color_idx], dtype=torch.long)
            
            # Generate
            z = enc(cap_tensor, color_tensor)
            out = gen(z)[0]
            
            # Denormalize
            out = (out * 0.5 + 0.5).clamp(0, 1)
            out = out.permute(1, 2, 0).cpu().numpy()
            out = (out * 255).astype(np.uint8)
            
            img = Image.fromarray(out)
            fname = f"sample_{i+1}_{cap.replace(' ', '_')}.png"
            img.save(results / fname)
            print(f"  ✓ {fname}")
    
    print(f"[✓] Saved to: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=".", help="Output directory")
    parser.add_argument("--make_dataset", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_per_combo", type=int, default=20)
    parser.add_argument("--samples", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.make_dataset:
        make_dataset(args.out_dir, num_per_combo=args.num_per_combo)
    
    if args.train:
        train(args)
    
    if args.generate:
        generate(args)
