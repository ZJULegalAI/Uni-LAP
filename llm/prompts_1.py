import json

def generate_prompt_version_choice(version,dataset):
    if dataset == "cail":
        if version == 'baseline':
            return generate_prompt_cail_baseline
        elif version == '2_stage':
            return generate_prompt_cail_step2
    
    elif dataset == "ecthr":
        if version == 'baseline':
            return generate_prompt_ecthr_baseline
        elif version == '2_stage':
            return generate_prompt_ecthr_step2


def generate_prompt_cail_baseline(sample_id, fact, candidate):
    """
    根据事实和小模型的预测结果生成大模型的 Prompt。
    :param sample_id: 当前案件的sample_id
    :param fact: 当前样本的事实
    :param small_model_prediction: 小模型对当前样本的预测法条比如["125", "133", "115"] 
    :param law_name: 法条具体解释
    :return: 生成的 Prompt 字符串
    """
    prompt = f"""# 请你完成一个根据案件事实进行法条预测的任务。

## 当前正在处理案件ID为：{sample_id}。当前事实案件是：{fact}

## 请从以下刑法法条候选集中进行法条的选取，实现对该事实案件的法条预测：{candidate}

## 请返回当前案件ID与你认为最相关的法条序号列表。具体格式要求如下：
{{ "案件ID": {sample_id}, "预测法条": ["125", "133", "115"] }}
""".strip()
    return prompt


def generate_prompt_ecthr_baseline(sample_id, fact):
    prompt = f"""# Please complete a task of predicting legal provisions based on the facts of the case.

## The current case ID being processed is:{sample_id}。The current factual case is:{fact}

## Based on your understanding of the case, please choose from the optional legal provisions ["ECHR Article 2", "ECHR Article 3", "ECHR Article 5", "ECHR Article 6", "ECHR Article 8", "ECHR Article 9", "ECHR Article 10", "ECHR Article 11", "ECHR Article 14", "ECHR Article 1 of Protocol 1"].

## requirement:
* The final predicted rule needs to be mapped to the sequence number, {{"ECHR Article 2":"0","ECHR Article 3":"1","ECHR Article 5":"2","ECHR Article 6":"3","ECHR Article 8":"4","ECHR Article 9":"5","ECHR Article 10":"6","ECHR Article 11":"7","ECHR Article 14":"8","ECHR Article 1 of Protocol 1":"9"}}
* The specific format requirements are as follows: {{"CaseID": {sample_id},  "Prediction_Law": ["3", "0", "1", "4", "6"] }}

""".strip()
    return prompt




def gen_legal_analysis(sample_id, fact,small_model_pred_law_index, law_name):

    analysis_template = f'''{{"案件ID": {sample_id},"分析结果": [{{"法条": "234", "分析": "步骤一：提取关键要素：1.行为条件：xxxx2.行为命令：xxxx3.行为构成要件：xxx步骤二：与案件事实对比：1.原文段落召回：xxx2.行为条件对比：xxxx3.行为命令对比：xxxx4.行为构成要件对比：xxx步骤三：评估法条适用性：xx", "分析结论": "匹配"}},{{"法条": "141", "分析": "步骤一：提取关键要素：1.行为条件：xxxx2.行为命令：xxxx3.行为构成要件：xxx步骤二：与案件事实对比：1.原文段落召回：xxx2.行为条件对比：xxxx3.行为命令对比：xxxx4.行为构成要件对比：xxx步骤三：评估法条适用性：xx", "分析结论": "不匹配"}}]}}
'''.strip()


    prompt = f"""# 请你完成一个基于法条对案件事实的适配分析的任务。

## 当前正在处理案件ID为：{sample_id}，当前事实案件是：{fact}

## 当前需要分析的{len(small_model_pred_law_index)}个刑法法条：{small_model_pred_law_index}，具体法条名称与内容：{law_name}。

## 请你按照以下步骤，逐一分析每个法条，请返回当前案件ID与分析结果：

步骤一： **提取关键要素**
从法条中提取以下关键要素：
1. 行为条件：描述适用法条的先决条件，涵盖行为主体的资格和发生行为的具体情境。
2. 行为命令：明确行为主体应遵循的规定，分为“必须行为”、“禁止行为”或“许可行为”。
3. 行为构成要件：指定法条所规定的具体行为模式或事件。

步骤二. **与案件事实对比**
1. 原文段落召回：请根据案件事实的相关内容，找出与“行为条件”“行为命令”“行为构成要件”三个关键要素相关的文本片段
2. 将三个关键要素相关的文本片段分别与法条相对应的关键要素进行对比，具体如下：
 - 行为条件对比：对比案件事实和法条的行为条件，判断事实是否属于法条定义的情境。
 - 行为命令对比:检查案件事实中是否存在法条所禁止、要求或允许的行为。
 - 行为构成要件对比：核实案件事实中的行为是否构成法条中的行为要件。

步骤三. **评估法条适用性**
结论只能是“匹配”和“不匹配”其中之一。请按照下面要求评估法条的适用性：
 - 全面匹配：如果案件事实满足法条的所有条件（行为条件、行为命令、行为构成要件），则该法条适用。
 - 不匹配或不完全匹配：如果案件事实中存在关键要素与法条不符，判断该法条不适用。

 ## 要求：
 * 法条名称与内容中当遇到一条法条含有多个罪名或者多个法条内容的情况，案件事实与其匹配时只需要满足其中一项罪名或法条内容即可。
 * 缺乏直接或间接证据的事实描述均假设为已成立。
 * 所有分析过程应基于语义匹配，而非仅限于字面一致。
 * 返回具体格式要求如下：{analysis_template}
 * 返回内容中不需要返回任何空格或换行符。

 """.strip()
    return prompt



# # Ours的
def generate_prompt_cail_step2(sample_id,  fact, analysis_results,small_model_prediction,law_name,true_label):
    """
    根据事实和小模型的预测结果生成大模型的 Prompt。
    :param sample_id: 当前案件的sample_id
    :param fact: 当前样本的事实
    :param small_model_prediction: 小模型对当前样本的预测法条比如["125", "133", "115"] 
    :param law_name: 法条具体解释
    :return: 生成的 Prompt 字符串
    """
    if analysis_results is not None:
        analysis_law_name = [item.get("") for item in analysis_results if item.get("分析结论") == "匹配"]
    else:
        analysis_law_name = []  # 如果 analysis_results 是 None，返回空列表

    prompt = f"""# 请你完成一个根据案件事实与法律分析结果进行刑法法条预测的任务。

## 当前正在处理案件ID为：{sample_id}。案件事实为：{fact}

## 请依次从以下可供参考的法条中选择认为最相关的{len(true_label)}条法条序号，有且仅选择{len(true_label)}个法条。
### 首先，经过法条与案件事实匹配得出，以下法条是最匹配的：{analysis_law_name},具体法律分析结果如下：{analysis_results}。
### 其次，考虑法条预测小模型提供的候选集：{small_model_prediction}，具体法条名称与内容：{law_name}
### 最后，考虑所有刑法法条。

## 请返回当前案件ID与你认为最相关的{len(true_label)}个法条组成的法条序号列表。

## 要求：
* 当存在相似法条且你多选一选择困难时，选择适用范围更窄、更具体的法条。
* 在适用范围相同时，选择{small_model_prediction}中顺序靠前的法条。
* 法条名称与内容中当遇到一条法条含有多个罪名或者多个法条内容的情况，案件事实与其匹配时只需要满足其中一项罪名或法条内容即可。
* 如果你认为正确法条是“某条之一”，如第133条之一，请直接返回"133"，而不是"133之一"。
* 具体格式要求如下：{{ "案件ID": {sample_id}, "预测法条": ["125", "133", "115"]}}

""".strip()
    return prompt




# Ours的
def gen_legal_analysis_ecthr(sample_id, fact,small_model_pred_law_index, law_name):

    analysis_template = f'''{{"CaseID": {sample_id},"Analysis_results": [{{"Article": "ECHR Article 2", "Analysis": "Step 1: 1. Behavioral Conditions:xxxx2. Behavioral Commands:xxxx3. Behavioral Constitutive Requirements: xxxStep 2: Evaluate the applicability of legal articles：xx", "Conclusion": "Match"}}, {{"Article": "ECHR Article 3", "Analysis": "Step 1: 1. Behavioral Conditions:xxxx2. Behavioral Commands:xxxx3. Behavioral Constitutive Requirements: xxxStep 2: Evaluate the applicability of legal articles：xx", "Conclusion": "Not Match"}}]}}
'''.strip()


# ECHR法条适用的正确标准是：
#  • 案件涉及的核心权利是否与该条款的保护范围一致。
#  • 案件事实是否触发了该条款所规定的行为条件（行为构成要件）。
#  • 即使最终裁决不支持侵权主张，只要相关权利受审查，仍可预测适用。

    prompt = f"""# Please complete a task of analyzing the alignment between legal articles and a factual case.

## Processing CaseID: {sample_id}，Factual case:{fact}

## The {len(small_model_pred_law_index)} ECTHR (European Court of Human Rights) articles to be analyzed: {small_model_pred_law_index}, specific article name and content: {law_name}.

## Please analyze each legal article saccording to the following instructions, and return the current CaseID along with the analysis results:

# Step 1:
# 1. Behavioral Conditions: Evaluate whether the case facts meet the prerequisites specified in the legal article.
# 2. Behavioral Comments: Determine if the case involves actions that align with the article’s mandatory, prohibited, or permissible requirements.
# 3. Behavioral Constitutive Requirements: Verify if the factual elements fulfill the essential criteria for the article’s applicability.

# Step 2:
# The conclusion is either “Match” or “Not Match” Please assess the applicability of the legal article based on the following criteria:
# -	Match: If the factual case satisfy conditions of the legal article, including Behavioral Conditions, Behavioral Comments, and Behavioral Constitutive Requirements, then the article is applicable. 
# -	Not Match or Incomplete Match: If key elements in the factual case do not align with the legal article, conclude that the article is not applicable.

## Requirements：
* The return format must follow: {analysis_template}.
* The returned content should not include any spaces or line breaks.


 """.strip()
    return prompt


# # Ours的
def generate_prompt_ecthr_step2(sample_id,  fact, analysis_results,small_model_prediction,law_name,true_label):
    """
    根据事实和小模型的预测结果生成大模型的 Prompt。
    :param sample_id: 当前案件的sample_id
    :param fact: 当前样本的事实
    :param small_model_prediction: 小模型对当前样本的预测法条比如["125", "133", "115"] 
    :param law_name: 法条具体解释
    :return: 生成的 Prompt 字符串
    """
    # if analysis_results is not None:
    #     analysis_law_name = [item.get("Article") for item in analysis_results if item.get("Analysis_conclusion") == "Match"]
    # else:
    #     analysis_law_name = []  # 如果 analysis_results 是 None，返回空列表

    prompt = f"""# Please complete a task of predicting ECTHR (European Court of Human Rights) articles based on a factual case and legal analysis results.
## Processing CaseID: {sample_id}, Factual case:{fact}

##The following are the predicted article numbers provided by the legal article prediction model as the candidate set ({len(small_model_prediction)} in total), ranked in descending order of predicted probability. Based on your understanding of the case, please select the {len(true_label)} most relevant article numbers from the reference list below. You must select exactly {len(true_label)} articles. 
## The candidate set provided by the legal article prediction model: {small_model_prediction}, specific article names: {law_name}.

## The detailed legal analysis are as follows: {analysis_results}. 

## Please return the current CaseID and a list of the {len(true_label)} article numbers you consider most relevant.

## Requirements:
* If you believe that the candidate set does not contain the correct answer, please select from [“ECHR Article 2”, “ECHR Article 3”, “ECHR Article 5”, “ECHR Article 6”, “ECHR Article 8”, “ECHR Article 9”, “ECHR Article 10”, “ECHR Article 11”, “ECHR Article 14”, “ECHR Article 1 of Protocol 1”]. Return the current case ID and a list of the {len(true_label)} article numbers you consider most relevant.
* If the required number of articles to predict ({len(true_label)}) exceeds the candidate set size ({len(small_model_prediction)}), please select the remaining ones from the available articles. Return the current case ID and the list of the most relevant article numbers.
* The final predicted legal articles need to be mapped to their corresponding numbers，{{"ECHR Article 2":"0","ECHR Article 3":"1","ECHR Article 5":"2","ECHR Article 6":"3","ECHR Article 8":"4","ECHR Article 9":"5","ECHR Article 10":"6","ECHR Article 11":"7","ECHR Article 14":"8","ECHR Article 1 of Protocol 1":"9"}}
* The specific format requirements are as follows: {{ "CaseID": {sample_id}, "Prediction_Law": ["3", "0", "1"] }}

""".strip()
    return prompt
