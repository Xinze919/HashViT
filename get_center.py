import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

def generate_semantic_proxies(class_names, bits=[16, 32, 48, 64], save_dir="/data4/liuxinze/Hash/DeepHash-pytorch/nuswide"):
    print("加载 CLIP 文本模型...")
    # 这里用 huggingface 的开源 CLIP，自动下载
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder.eval()

    # 1. 构造 Prompt 增强语义 (这是个常用小 Trick)
    prompts = [f"a photo of a {name}" for name in class_names]
    
    # 2. 提取 512 维的文本特征
    inputs = tokenizer(prompts, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        # 获取 [CLS] token 的特征，形状: [num_classes, 512]
        text_embeddings = outputs.last_hidden_state[:, 0, :] 
    
    # 归一化，得到纯粹的余弦语义空间
    text_embeddings = nn.functional.normalize(text_embeddings, p=2, dim=-1)

    # 3. 映射到不同的 bit 维度，并保存
    for bit in bits:
        # 为了完美保留单词之间的余弦距离比例，我们使用高斯正交投影矩阵
        # Johnson-Lindenstrauss 引理保证了这种映射能最大程度保留拓扑
        projection_matrix = torch.empty(512, bit)
        nn.init.orthogonal_(projection_matrix)
        
        # 映射到哈希维度
        hash_proxies = torch.mm(text_embeddings, projection_matrix)
        
        # 再次归一化，推到超球面上
        hash_proxies = nn.functional.normalize(hash_proxies, p=2, dim=-1)
        
        # 保存为 PT 文件
        save_path = f"{save_dir}/nuswide_text_proxies_{bit}bit.pt"
        torch.save(hash_proxies.clone().detach(), save_path)
        print(f"成功保存 {bit} bit 的文本语义中心 -> {save_path}")

if __name__ == "__main__":
    # 这里替换成 NUS-WIDE 81 的真实纯英文类别名列表
    nuswide_81_classes = [
    "airport", "animal", "beach", "bear", "birds", "boats", "book", "bridge", "buildings", "cars",
    "castle", "cat", "cityscape", "clouds", "computer", "coral", "cow", "dancing", "dog", "earthquake",
    "elk", "fire", "fish", "flags", "flowers", "food", "fox", "frost", "garden", "glacier",
    "grass", "gull", "harbor", "horses", "house", "lake", "leaf", "map", "military", "moon",
    "mountain", "nighttime", "ocean", "person", "plants", "police", "protest", "railroad", "rainbow", "reflection",
    "road", "rocks", "running", "sand", "sign", "sky", "snow", "soccer", "sports", "statue",
    "street", "sun", "sunset", "surf", "swimmers", "tattoo", "temple", "tiger", "tower", "town",
    "toy", "train", "tree", "valley", "vehicle", "water", "waterfall", "wedding", "whales", "window",
    "zebra"
    ]
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer", 
        "dog", "frog", "horse", "ship", "truck"
    ]
    # generate_semantic_proxies(nuswide_81_classes, save_dir='/data4/liuxinze/Hash/DeepHash-pytorch/nuswide')
    # generate_semantic_proxies(nuswide_81_classes, save_dir='/data4/liuxinze/Hash/DeepHash-pytorch/nuswide')
    # generate_semantic_proxies(cifar10_classes, save_dir='/data4/liuxinze/Hash/DeepHash-pytorch/imagenet')
    # generate_semantic_proxies(cifar10_classes, save_dir='/data4/liuxinze/Hash/DeepHash-pytorch/cifar')