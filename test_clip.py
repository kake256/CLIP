import torch
import clip
from PIL import Image

# GPUが利用可能かチェックし、利用可能ならGPU(cuda)を、そうでなければCPUを使用します
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIPモデル(ViT-B/32)をロードします
print("CLIPモデルをロード中...")
model, preprocess = clip.load("ViT-B/32", device=device)
print("モデルのロード完了。")

# ダウンロードした画像を読み込み、モデルが処理できる形式に変換します
img_path = "CLIP/img/CLIP.png"
image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

# 比較したいテキストの候補をリストで用意します
text_descriptions = ["a diagram", "a dog", "a cat", "a figure", "a bird"]
text_tokens = clip.tokenize(text_descriptions).to(device)

# モデルの計算を実行します
with torch.no_grad():
    # 画像と各テキストの類似度を計算
    logits_per_image, logits_per_text = model(image, text_tokens)
    # 類似度を確率に変換します
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 結果を表示します
print("\n--- 判定結果 ---")
for i, description in enumerate(text_descriptions):
    print(f"'{description}' である確率: {probs[0, i]*100:.2f}%")