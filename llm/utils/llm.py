def prompts_to_chat_messages(prompts):
    if isinstance(prompts, str):
        prompts = [prompts]
    all_messages = [prompt_to_chat_message(prompt) for prompt in prompts]

    return all_messages


def prompt_to_chat_message(prompt):
    # 如果prompt是一个字符串，那么将其转换为一个包含单个用户角色消息的列表
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    
    elif isinstance(prompt, list):
        chat_messages = []

        # 如果列表prompt的长度是偶数，那么将列表的第一个元素作为一个系统角色消息添加到chat_messages中，并将prompt缩短为除去第一个元素的列表。
        if len(prompt) % 2 == 0:
            chat_messages.append({"role": "system", "content": prompt[0]})
            prompt = prompt[1:]
        
        # 根据索引的奇偶性分配用户或助手角色，并将每个内容作为一个消息添加到chat_messages列表中。
        for i, content in enumerate(prompt):
            role = "user" if i % 2 == 0 else "assistant"
            chat_messages.append({"role": role, "content": content})

        return chat_messages
    
    elif prompt is None:
        return None
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")
