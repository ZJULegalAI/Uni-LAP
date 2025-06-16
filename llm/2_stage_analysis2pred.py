import json
import re

# 读取JSON文件
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 提取法条序号
def extract_law_number(law_text):
    if not law_text:
        return None
    # 使用正则表达式匹配序号
    match = re.search(r'\d+', law_text)
    return match.group(0) if match else None

# 提取分析结果为“匹配”的法条序号
def extract_matched_laws(data):
    matched_laws = []
    for item in data:
        if item and isinstance(item, list):
            # 提取当前条目中所有匹配的法条序号
            matched_numbers = []
            for law_item in item:
                if isinstance(law_item, dict) and law_item.get("分析结论") == "匹配":
                    law_number = extract_law_number(law_item.get("法条"))
                    if law_number:
                        matched_numbers.append(law_number)
            # 如果当前条目有匹配的法条，则写入列表；否则写入None
            matched_laws.append(matched_numbers if matched_numbers else None)
        else:
            # 如果item是null或不符合预期结构
            matched_laws.append(None)
    return matched_laws

# 转换为类似第二张图的JSON格式
def convert_to_output_format(matched_laws):
    return [None if law is None else law for law in matched_laws]

# 主函数
def main(input_file_path, output_file_path):
    # 读取输入JSON文件
    data = read_json_file(input_file_path)
    
    # 提取匹配的法条序号
    matched_laws = extract_matched_laws(data)
    
    # 转换为目标格式
    output_json = convert_to_output_format(matched_laws)
    
    # 将结果写入输出文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    
    print(f"结果已写入文件: {output_file_path}")

# 示例调用
if __name__ == "__main__":
    input_file_path = "/home/u22451152/Uni-LAP/llm/results/cail/crime-bert/GPT-4o/llm_analysis_results.json"  # 输入JSON文件路径
    output_file_path = "/home/u22451152/Uni-LAP/llm/results/cail/crime-bert/GPT-4o/step1_pred_result.json"  # 输出JSON文件路径
    main(input_file_path, output_file_path)