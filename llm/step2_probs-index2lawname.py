import json
import os

topk = 3

# CAIL
# step1小模型给的结果
# law_probs_path = "/home/u22451152/Uni-LAP/llm/datasets/cail/Probs/Crime-Bert_probs_not_topk/law_probs.json"
# law_index2name_path = "/home/u22451152/Uni-LAP/llm/datasets/cail/data/law_labels_cail_filtered.txt"
# top_k_path = f"/home/u22451152/Uni-LAP/llm/datasets/cail/Probs/Crime-Bert_probs_not_topk"

# ECTHR
# step1小模型给的结果
law_probs_path = "/home/u22451152/Uni-LAP/llm/datasets/ecthr/Probs/Legal-Bert_probs_not_topk/law_probs.json"
law_index2name_path = "/home/u22451152/Uni-LAP/llm/datasets/ecthr/data/law_labels_ecthr_filtered.txt"
top_k_path = f"/home/u22451152/Uni-LAP/llm/datasets/ecthr/Probs/Legal-Bert_probs_not_topk"

if not os.path.exists(top_k_path):
    os.makedirs(top_k_path)

# 读取 JSON 文件
with open(law_probs_path, 'r', encoding='utf-8') as f:
    probability_lists = json.load(f)

# 读取 TXT 文件
with open(law_index2name_path, 'r', encoding='utf-8') as f:
    law_lines = f.readlines()

# 创建一个列表来存储所有样本的结果
all_samples_top_k_laws = []

# 获取法条的总数
num_labels = len(law_lines)

# 遍历 JSON 文件中的每组概率（每个样本）
for probabilities in probability_lists:
    # 创建一个列表来存储当前样本的法条序号和对应的概率
    law_probabilities = []
    
    # 遍历当前样本的概率值
    for idx, prob in enumerate(probabilities):
        # 检查索引是否在法条序号范围内
        if idx < num_labels:
            law_number = law_lines[idx].strip()  # 获取对应的法条序号并去除换行符
            law_probabilities.append({'law_number': law_number, 'probability': prob})
    
    # 按概率降序排序当前样本的法条
    sorted_law_probabilities = sorted(law_probabilities, key=lambda x: x['probability'], reverse=True)
    
    # 取前k个概率最大的法条序号（不需要概率值）
    top_k_laws = [item['law_number'] for item in sorted_law_probabilities[:topk]]
    
    # 将当前样本的前k个法条序号添加到结果列表中
    all_samples_top_k_laws.append(top_k_laws)

# 保存到新的 JSON 文件
with open(f"{top_k_path}/law_name_top{topk}.json", 'w', encoding='utf-8') as f:
    json.dump(all_samples_top_k_laws, f, ensure_ascii=False, indent=4)

print(f"Top {topk} laws for all samples have been saved {top_k_path}.")