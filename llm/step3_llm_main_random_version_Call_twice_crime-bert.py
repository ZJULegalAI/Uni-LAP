import json
import os
import random
from llm.llm_api import APIChat
from prompts_1 import generate_prompt_version_choice, gen_legal_analysis
from utils.utils import extract_and_parse_json, law_idx2name, get_ecthr_law_name
import pickle

llm = APIChat(
    model='',  # llm模型
    api_key='', # api-key
    base_url='',  # 中转url
    num_workers=16,  # 并发 worker 数量
)


def load_processed_indices(file_path):
    """加载已处理的样本序号"""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return set(json.load(f))
    return set()

def save_processed_indices(file_path, indices):
    """保存已处理的样本序号"""
    with open(file_path, 'w') as f:
        json.dump(list(indices), f)

def load_existing_results(file_path, total_samples):
    """加载已有的结果文件，如果不存在则初始化一个全为 None 的列表"""
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return [None] * total_samples

def save_results(file_path, results):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def save_bad_case(file_path, sample_id, fact,law_analysis,  true_label, predicted_laws, candidate_laws):
    with open(file_path, 'a', encoding='utf-8') as f: 
        f.write(f"样本 ID: {sample_id}\n")
        f.write(f"事实描述: {fact}\n")
        f.write(f"法律分析: {law_analysis}\n")
        f.write(f"真实标签: {true_label}\n")
        f.write(f"大模型预测标签: {predicted_laws}\n")
        f.write(f"小模型候选法条: {candidate_laws}\n")
        f.write("-" * 50 + "\n\n")


def main():
    # 参数设置  TODO dataset
    prompt_version = {"prompt":"2_stage", "dataset":"cail"} # dataset:[]
    # step2得到的结果 law_name_top3.json  TODO
    small_model_top_k_laws_path = "/home/u22451152/Uni-LAP/llm/datasets/cail/Probs/Crime-Bert_probs/law_name_top3.json"
    # 存结果的地址 TODO
    llm_results_path = "/home/u22451152/Uni-LAP/llm/results/cail/crime-bert/gpt-3.5-turbo"  # 保存结果

    data_path = f"/home/u22451152/Uni-LAP/llm/datasets/{prompt_version['dataset']}/data"
    # 测试数据  TODO
    test_dataset_path = os.path.join(data_path, "raw_pkl/test_dataset.pkl")
    # 法条名称 
    index2name_path = os.path.join(data_path, "law_name_define_0124.txt")
    law_index2lawdefinition_path = os.path.join(data_path, "law_labels_cail_filtered.txt")
    
    # 保存已处理样本序号的文件
    processed_indices_path = os.path.join(llm_results_path, "processed_indices.json")
    # 保存大模型预测结果的文件
    results_file_path = os.path.join(llm_results_path, "llm_pred_results.json")
    # 保存法律分析结果的文件
    analysis_file_path = os.path.join(llm_results_path, "llm_analysis_results.json")

    if not os.path.exists(llm_results_path):
        os.makedirs(llm_results_path)

    # 读取 JSON 文件（小模型的预测结果）
    with open(small_model_top_k_laws_path, 'r', encoding='utf-8') as f:
        small_model_predictions = json.load(f)

    # 读取 PKL 文件（包含事实和法条标签）
    with open(test_dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    # 读取 TXT 文件
    with open(law_index2lawdefinition_path, 'r', encoding='utf-8') as f:
        law_name_lines = f.readlines()
    
    with open(index2name_path, 'r', encoding='utf-8') as f:
        law_difine_lines = f.readlines()  # 读取所有行
    
    # 保存 bad case 的文件路径
    bad_case_file_path = os.path.join(llm_results_path, "bad_cases.txt")
    if not os.path.exists(bad_case_file_path):
        with open(bad_case_file_path, 'w', encoding='utf-8') as f:
            f.write("Bad Cases 记录\n")
            f.write("=" * 50 + "\n\n")

    # 检查数据长度是否一致
    if len(small_model_predictions) != len(data['raw_facts_list']):
        raise ValueError("小模型预测结果和测试集样本数量不一致！")

    # 获取生成prompt的函数
    gen_prompt = generate_prompt_version_choice(prompt_version["prompt"], prompt_version["dataset"])

    # 加载已有的结果文件
    all_ans_list = load_existing_results(results_file_path, len(small_model_predictions))
    all_analysis_list = load_existing_results(analysis_file_path, len(small_model_predictions))

    # 加载剩下的序号
    processed_indices = load_processed_indices(processed_indices_path)
    all_indices = set(range(len(small_model_predictions)))
    remaining_indices =  list(all_indices - processed_indices)


    # 随机选择 100 个样本
    total_size = 300  # TODO
    random_size = min(total_size, len(remaining_indices))
    random_indices = random.sample(remaining_indices, min(total_size, len(remaining_indices)))
    processed_indices.update(random_indices)
    print("random_indices:",random_indices)
    
    batch_size = 50  # 每批 batch_size 个元素

    # 生成法律分析的prompt
    step1_tasks = []
    for i in random_indices:
        sample_id = i
        fact = data['raw_facts_list'][i]  # 当前样本的事实
        small_model_pred_law_index = small_model_predictions[i]  # 当前样本的小模型预测法条

        # 获得法条定义
        law_name = law_idx2name(prompt_version["dataset"], law_difine_lines, small_model_pred_law_index)
        if prompt_version['dataset'] == 'ecthr':
            small_model_pred_law_index = get_ecthr_law_name(law_difine_lines, small_model_pred_law_index)

        prompt = gen_legal_analysis(sample_id, fact,small_model_pred_law_index, law_name)  # 生成 Prompt
        print("生成prompt：\n\n", prompt)
        step1_tasks.append(prompt)
    

    for i in range(0, random_size, batch_size):
        batch_task = step1_tasks[i:i + batch_size]
        batch_responses = llm(batch_task, temperature=0.2)   # 调用api
        print(f"batch {i} 调用api后：", batch_responses)

        # 解析JSON响应
        batch_results = [extract_and_parse_json(response, "{}") for response in batch_responses if response is not None]
        print(f"batch {i} 解析json后：", batch_results)

        for item in batch_results:
            if item is None: 
                print("Warning: 跳过空响应")
                continue 
            
            sample_id = item["案件ID"]
            law_analysis = item["分析结果"]

            ################################## 打印看看 #####################################

            fact = data['raw_facts_list'][sample_id]
            candidate_laws = small_model_predictions[sample_id]

            print("step1：")
            print(f"样本 ID: {sample_id}")
            # print(f"事实描述: {fact}")
            print(f"大模型法律分析结果: {law_analysis}")
            print(f"小模型候选法条: {candidate_laws}") 
            print("-" * 50) 

            #######################################################################3

            # 确保 law_analysis 是一个列表
            if isinstance(law_analysis, str):
                try:
                    law_analysis = json.loads(law_analysis)  # 尝试将字符串转换为列表
                except json.JSONDecodeError:
                    print(f"Warning: 无法解析 law_analysis 为 JSON，样本 ID 为 {sample_id}")
                    law_analysis = []  # 解析失败时，将其设置为空列表

            # 如果 law_analysis 是空字符串或 None，也设置为空列表
            if law_analysis in ["", None]:
                law_analysis = []
            
            all_analysis_list[sample_id] = law_analysis

        print(f"样本 前{i + batch_size}/{len(step1_tasks)} 处理完成。")


    # 生成预测结果的prompt
    step2_tasks = []
    for i in random_indices:
        sample_id = i

        law_analysis = all_analysis_list[i]
        fact = data['raw_facts_list'][i]

        # TODO  law_analysis如果是空 特殊处理
        if law_analysis is not None:
            matched_law_analysis = [item for item in law_analysis if item.get("分析结论") == "匹配"]
        else:
            print("law_analysis：", law_analysis)
            matched_law_analysis = []  # 如果 law_analysis 是 None，返回空列表

        true_label = [law_name_lines[key].strip() for key in data['law_label_lists'][i]]
        small_model_pred_law_index = small_model_predictions[i]  # 当前样本的小模型预测法条

        # 获得法条定义
        law_name = law_idx2name(prompt_version["dataset"], law_difine_lines, small_model_pred_law_index)
        if prompt_version['dataset'] == 'ecthr':
            small_model_pred_law_index = get_ecthr_law_name(law_difine_lines, small_model_pred_law_index)

        prompt = gen_prompt(sample_id, fact, matched_law_analysis, small_model_pred_law_index, law_name,true_label)  # 生成 Prompt
        print("生成prompt：\n\n", prompt)
        print("真实标签是：", true_label)
        step2_tasks.append(prompt)

    total_error_count = 0
    for i in range(0, random_size, batch_size):
        batch_task = step2_tasks[i:i + batch_size]
        batch_responses = llm(batch_task, temperature=0.2)   # 调用api
        print(f"batch {i} 调用api后：", batch_responses)

        # 解析JSON响应
        batch_results = [extract_and_parse_json(response, "{}") for response in batch_responses if response is not None]
        print(f"batch {i} 解析json后：", batch_results)

        batch_error_count = 0
        for item in batch_results:
            if item is None:  # 如果有空响应
                print("Warning: 跳过空响应")
                continue  # 跳过空响应
            
            sample_id = item["案件ID"]
            predicted_laws = item["预测法条"]

            ################################## 找bad case #####################################

            fact = data['raw_facts_list'][sample_id]
            law_analysis = all_analysis_list[sample_id]
            true_label = [law_name_lines[key].strip() for key in data['law_label_lists'][sample_id]]
            candidate_laws = small_model_predictions[sample_id]

            if set(predicted_laws) != set(true_label):  # 预测错误
                batch_error_count += 1
                print("预测错误的样本：")
                print(f"样本 ID: {sample_id}")
                print(f"事实描述: {fact}")
                print(f"法律分析: {law_analysis}")
                print(f"真实标签: {true_label}")
                print(f"大模型预测标签: {predicted_laws}")
                print(f"小模型候选法条: {candidate_laws}") 
                print("-" * 50) 
            
                # 将 bad case 保存到文件中
                save_bad_case(bad_case_file_path, sample_id, fact, law_analysis, true_label, predicted_laws, candidate_laws)

            #######################################################################3

            # 确保 predicted_laws 是一个列表
            if isinstance(predicted_laws, str):
                try:
                    predicted_laws = json.loads(predicted_laws)  # 尝试将字符串转换为列表
                except json.JSONDecodeError:
                    print(f"Warning: 无法解析 predicted_laws 为 JSON，样本 ID 为 {sample_id}")
                    predicted_laws = []  # 解析失败时，将其设置为空列表

            # 如果 predicted_laws 是空字符串或 None，也设置为空列表
            if predicted_laws in ["", None]:
                predicted_laws = []
            
            all_ans_list[sample_id] = predicted_laws

        print(f"本batch预测错误的总样本数：{batch_error_count} / {batch_size}")
        total_error_count += batch_error_count
        print(f"样本 前{i + batch_size}/{len(step2_tasks)} 处理完成。")
    
    print(f"所有预测错误的总样本数：{total_error_count} / {random_size}")

    # 保存所有样本的大模型预测结果
    save_results(results_file_path, all_ans_list)
    save_results(analysis_file_path, all_analysis_list)

    # 保存已处理的样本序号
    save_processed_indices(processed_indices_path, processed_indices)

    print(f"所有样本的大模型预测结果已保存到 {results_file_path}")
    print(f"所有样本的法律分析结果已保存到 {analysis_file_path}")

if __name__ == "__main__":
    main()