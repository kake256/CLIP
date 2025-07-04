import torch
import clip
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# GPUが利用可能ならGPUを、そうでなければCPUを使用
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIPモデルと、モデルに合わせた画像前処理をロード
model, preprocess = clip.load('ViT-B/32', device)

# CIFAR-10のテストデータセットをダウンロード・準備
# transform=preprocess を指定し、CLIPモデル用の前処理を自動で適用
test_dataset = CIFAR10(root="~/.cache", download=True, train=False, transform=preprocess)

def extract_features(dataset, model, device):
    all_features = []
    all_labels = []
    
    print("Extracting features from the dataset...")
    # 勾配計算を無効にし、メモリ効率を上げる
    with torch.no_grad():
        # tqdmでプログレスバーを表示
        for images, labels in tqdm(DataLoader(dataset, batch_size=500)):
            # model.encode_imageで特徴量を抽出
            features = model.encode_image(images.to(device))
            
            # 特徴量とラベルをリストに追加
            all_features.append(features)
            all_labels.append(labels)

    # リストを結合してNumPy配列に変換
    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

# 特徴量とラベルを抽出
test_features, test_labels = extract_features(test_dataset, model, device)

print("Feature extraction complete.")
print("Shape of features:", test_features.shape)
print("Shape of labels:", test_labels.shape)

print("\nRunning t-SNE... This may take a few minutes.")

# t-SNEを初期化
# n_components=2 で2次元に削減
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)

# 特徴量を2次元に圧縮
features_2d = tsne.fit_transform(test_features)

print("t-SNE complete.")

# 可視化の準備
plt.figure(figsize=(16, 12))

# seabornのscatterplotを使って、クラスごとに色分けしてプロット
sns.scatterplot(
    x=features_2d[:, 0], 
    y=features_2d[:, 1],
    hue=[test_dataset.classes[l] for l in test_labels], # ラベル名で色分け
    palette='tab10', # 10色のカラーパレット
    legend='full'
)

plt.title('t-SNE Visualization of CLIP Features on CIFAR-10', fontsize=16)
plt.xlabel('t-SNE Dimension 1', fontsize=12)
plt.ylabel('t-SNE Dimension 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # 凡例をグラフの外に表示
plt.grid(True)

# ------------------- 変更箇所 -------------------
# plt.show() # 画面表示の代わりにファイルに保存する

# グラフをPDFファイルとして保存
output_filename = "cifar10_tsne_visualization.pdf"
plt.savefig(output_filename, bbox_inches='tight')

print(f"\nプロットを '{output_filename}' として保存しました。")
# ----------------------------------------------