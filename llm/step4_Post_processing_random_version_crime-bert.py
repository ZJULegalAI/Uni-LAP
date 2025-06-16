import json
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

# 读取JSON文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 读取TXT文件
def read_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# 读取PKL文件
def read_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 计算指标
def calculate_metrics(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    mf1 = f1_score(true_labels, pred_labels, average='macro')
    mp = precision_score(true_labels, pred_labels, average='macro')
    mr = recall_score(true_labels, pred_labels, average='macro')
    return acc, mf1, mp, mr

# 保存结果到文件
def save_results(results, file_path):
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

# 读取历史结果
def load_results(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "accuracy": [],
            "macro_f1": [],
            "macro_precision": [],
            "macro_recall": []
        }  # 如果文件不存在，返回空结果

# 绘制结果图
def plot_results(results):
    plt.figure(figsize=(12, 8))

    # 绘制 Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(results["accuracy"], marker='o', label="Accuracy")
    plt.xlabel("Run")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Runs")
    plt.legend()

    # 绘制 Macro F1
    plt.subplot(2, 2, 2)
    plt.plot(results["macro_f1"], marker='o', label="Macro F1")
    plt.xlabel("Run")
    plt.ylabel("Macro F1")
    plt.title("Macro F1 over Runs")
    plt.legend()

    # 绘制 Macro Precision
    plt.subplot(2, 2, 3)
    plt.plot(results["macro_precision"], marker='o', label="Macro Precision")
    plt.xlabel("Run")
    plt.ylabel("Macro Precision")
    plt.title("Macro Precision over Runs")
    plt.legend()

    # 绘制 Macro Recall
    plt.subplot(2, 2, 4)
    plt.plot(results["macro_recall"], marker='o', label="Macro Recall")
    plt.xlabel("Run")
    plt.ylabel("Macro Recall")
    plt.title("Macro Recall over Runs")
    plt.legend()

    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 文件路径
    # step3得到的结果
    pred_json_file = '/home/u22451152/Uni-LAP/llm/results/cail/crime-bert-not-topk/GPT-4o_v3/llm_pred_results.json'
    label_txt_file = '/home/u22451152/Uni-LAP/llm/datasets/cail/data/law_labels_cail_filtered.txt'
    test_datasets_pkl_file = '/home/u22451152/Uni-LAP/llm/datasets/cail/data/raw_pkl/test_dataset.pkl'
    results_dic = '/home/u22451152/Uni-LAP/llm/results/cail/crime-bert-not-topk/GPT-4o_v3'  # 保存结果的文件
    results_file = os.path.join(results_dic, "final_results.json")
    # 读取数据
    pred_json_data = read_json(pred_json_file)
    label_txt_data = read_txt(label_txt_file)
    test_datasets = read_pkl(test_datasets_pkl_file)

    # 获取真实标签
    true_labels_ori = test_datasets['law_label_lists']  # n个list
    num_label = len(label_txt_data)

    # 初始化 true_labels 和 pred_labels
    true_labels = []
    pred_labels = []

    # 遍历每个样本
    for i, sample in enumerate(pred_json_data):
        if sample is None:
            continue  # 跳过 null 样本

        # 生成预测标签
        pred = [0] * num_label
        for item in sample:
            if item in label_txt_data:
                idx = label_txt_data.index(item)
                pred[idx] = 1
        pred_labels.append(pred)

        # 生成真实标签
        true = [0] * num_label
        for law_id in true_labels_ori[i]:
            true[law_id] = 1
        true_labels.append(true)

    # 计算指标
    acc, mf1, mp, mr = calculate_metrics(true_labels, pred_labels)

    # 输出当前结果
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {mf1:.4f}")
    print(f"Macro Precision: {mp:.4f}")
    print(f"Macro Recall: {mr:.4f}")

    # 加载历史结果
    results = load_results(results_file)

    # 添加当前结果
    results["accuracy"].append(acc)
    results["macro_f1"].append(mf1)
    results["macro_precision"].append(mp)
    results["macro_recall"].append(mr)

    # 保存更新后的结果
    save_results(results, results_file)

    # 绘制结果图
    plot_results(results)

if __name__ == "__main__":
    main()