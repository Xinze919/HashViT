import numpy as np
from nltk.corpus import wordnet as wn

def extract_imagenet100_classes(txt_path):
    index_to_wnid = {}
    
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_path = parts[0]
            
            # 【注意】这里假设你的路径形如: xxx/n01440764/xxx.JPEG
            # 如果你的文件夹层级不同，请稍微修改 split 的索引！
            # 只要把形如 n0xxxxxxx 的字符串提取出来赋给 wnid 即可
            # wnid = img_path.split('/')[1]  
            wnid = img_path.split('/')[-1].split('_')[0]
            
            # 提取 100 维的 One-Hot 标签，找到 '1' 所在的索引
            label_vector = [float(x) for x in parts[1:]]
            class_idx = np.argmax(label_vector)
            
            # 记录 索引 -> wnid 的映射
            if class_idx not in index_to_wnid:
                index_to_wnid[class_idx] = wnid
                if len(index_to_wnid) == 100:
                    break # 100个类全找到了，提前结束
                    
    # 将 wnid 翻译成人类可读的英文单词
    imagenet100_classes = []
    # 严格按照 0 到 99 的顺序遍历，保证和你的 One-Hot 标签绝对对齐！
    for i in range(100):
        wnid = index_to_wnid[i]
        # 通过 wordnet 库将 n0 开头的 ID 转换为真实单词
        synset = wn.synset_from_pos_and_offset('n', int(wnid[1:]))
        # 取第一个同义词，并把下划线替换为空格
        class_name = synset.lemma_names()[0].replace('_', ' ')
        imagenet100_classes.append(class_name)

    print("\n提取成功！请直接复制以下列表到你的 CLIP 生成脚本中：\n")
    print("imagenet100_classes = [")
    for name in imagenet100_classes:
        print(f'    "{name}",')
    print("]")

if __name__ == "__main__":
    # 把这里换成你 ImageNet-100 的真实 train.txt 路径
    train_txt_path = "/data4/liuxinze/Hash/DeepHash-pytorch/data/imagenet/train.txt" 
    extract_imagenet100_classes(train_txt_path)