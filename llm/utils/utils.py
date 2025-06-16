import json


def extract_and_parse_json(input_string, wrapper):
    """
    尝试从字符串中提取并解析 JSON。

    :param input_string: 输入的字符串
    :param wrapper: JSON 的包裹符号，可以是 '{}' 或 '[]'
    :return: 解析后的 JSON 对象，如果解析失败则返回 None
    """
    if len(wrapper) != 2:
        raise ValueError("Wrapper must be exactly two characters long")

    start_char, end_char = wrapper
    start_index = input_string.find(start_char)
    end_index = input_string.rfind(end_char)

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        print("解析失败！",input_string)
        return None

    json_string = input_string[start_index:end_index + 1]

    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        print("解析失敗：",input_string)
        return None
    

def law_idx2name(dataset_version, law_lines, law_idx):
    if dataset_version == 'cail':
        # 将法条编号和名称存储为字典
        law_dict = {}
        for line in law_lines:
            if '：' in line:  # 确保行中包含分隔符
                parts = line.strip().split('：', 1)  # 按第一个冒号分割
                if len(parts) == 2:
                    law_number = parts[0].replace('刑法第', '').replace('条', '')  # 提取编号
                    law_name = parts[1]  # 提取名称
                    law_dict[law_number] = law_name

        # 根据 law_idx 查找对应的法条名称
        result = []
        for idx in law_idx:
            if idx in law_dict:
                result.append(f"刑法第{idx}条：{law_dict[idx]}")
            else:
                result.append(f"刑法第{idx}条：未知法条名称")  # 如果编号不存在，返回未知

        # 拼接结果并返回
        return '、 '.join(result) 
    elif dataset_version == 'ecthr':
        result = []
        for idx in law_idx:
            index = int(idx)
            result.append(law_lines[index].strip())
        return '、 '.join(result)
        
def get_ecthr_law_name(law_lines, index_list):

    law_name_list = []
    for line in law_lines:
        if ' - ' in line:  # 确保行中包含分隔符
            parts = line.strip().split(' - ', 1)  # 按第一个冒号分割
            if len(parts) == 2:
                law_number = parts[0]  # 提取编号
                law_name_list.append(law_number)

    result = []
    for idx in index_list:
        index = int(idx)
        result.append(law_name_list[index].strip())
    return result


def law_baseline(index2name_path):
    with open(index2name_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()  # 读取所有行
        # 去除每行的换行符并过滤空行
        content_list = [line.strip() for line in lines if line.strip()]
    return content_list