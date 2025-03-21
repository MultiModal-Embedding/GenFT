from datasets import load_dataset, Dataset
import os
import json
import random
import time
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_jsonl_dataset(file_path):
    data_list = []
    for i in range(10):
        file = f"{file_path}/doc_corpus_add_cases_data_embedding_0{i}"
        with open(file, "r") as f:
            lines = f.readlines()
            for j in range(len(lines)):
                line = lines[j].strip()
                if not line.endswith("}"):
                    if not line.endswith("]"):
                        if line.endswith(","):
                            line = line[:-1]
                        line += "]"
                    line += "}"
                doc_json = json.loads(line)
                data_list.append(doc_json)
    return Dataset.from_list(data_list)


def sample_dataset(dataset, sample_ratio=0.1, seed=42):
    random.seed(seed)
    sample_size = int(len(dataset) * sample_ratio)
    print(sample_size)
    indices = random.sample(range(len(dataset)), sample_size)
    return dataset.select(indices)


def llm_search_strategy(original_sentence, similarity_scores, vllm):
    prompt = [
        {
            "role": "system",
            "content": "你是一个负责生成相似句子的专家。你需要为给定的句子生成多个同义句，每个同义句的语义相似度必须精确匹配指定的值，并且在句式、词汇和长度上有所变化，以确保生成句的多样性。"
        },
        {
            "role": "user",
            "content": f"请为下面的句子生成多个同义句，每个生成句的语义相似度必须精确匹配以下值：{', '.join(map(str, similarity_scores))}。\n"
                    f"原句：{original_sentence}\n"
                    f"要求：\n"
                    f"1. 对于低相似度（0.1-0.5），生成句应在句式、词汇和长度上有较大变化。\n"
                    f"2. 对于高相似度（0.6-0.99），生成句应在保持语义相似的同时，引入更多句式、词汇或表达方式的变化，避免过于相似。\n"
                    f"3. 确保语义相似度更高的生成"
                    f"4. 返回结果必须为 JSON 格式，格式如下：\n"
                    f'[{{"generated_sentence": <同义句1>, "similarity": {similarity_scores[0]}}}, '
                    f'{{"generated_sentence": <同义句2>, "similarity": {similarity_scores[1]}}}, '
                    f'..., '
                    f'{{"generated_sentence": <同义句N>, "similarity": {similarity_scores[-1]}}}]'
        }
    ]

    try:
        response = vllm.generate(prompt)
        response = json.loads(response.text)[
            'choices'][0]['message']['content']

        if '```json' in response:
            response = response.split('```json')[-1].split('```')[0]

        results = json.loads(response)
        generated_pairs = []
        for result in results:
            generated_sentence = result.get("generated_sentence", "").strip()
            similarity = result.get("similarity", 0.0)
            generated_pairs.append({
                "sentence1": original_sentence,
                "sentence2": generated_sentence,
                "score": similarity
            })
        return generated_pairs
    except Exception as e:
        print(
            f"生成失败，句子: {original_sentence[:30]}... 错误: {e} response: {response}")
        return []


def mask_and_complete_strategy(original_sentence, similarity_scores, vllm):
    def mask_tokens(sentence, similarity):
        tokens = sentence.split()
        num_tokens = len(tokens)

        mask_ratio = 1.0 - similarity
        num_to_mask = max(1, min(int(num_tokens * mask_ratio), num_tokens - 1))
        mask_indices = random.sample(range(num_tokens), num_to_mask)

        masked_sentence = " ".join(
            ["[MASK]" if i in mask_indices else token for i, token in enumerate(tokens)])
        return masked_sentence

    generated_pairs = []
    for similarity in similarity_scores:
        masked_sentence = mask_tokens(original_sentence, similarity)
        prompt = [
            {
                "role": "system",
                "content": "你是一个负责补全句子的专家。你需要根据给定的 mask 句子生成一个完整且语义通顺的句子。"
            },
            {
                "role": "user",
                "content": f"请将下面的句子中的 [MASK] 替换为合适的词或短语，生成一个完整且语义通顺的句子。\n"
                        f"mask 句子：{masked_sentence}\n"
                        f"要求：\n"
                        f"1. 生成的句子应语义通顺且符合上下文。\n"
                        f"2. 返回结果必须为 JSON 格式，格式如下：\n"
                        f'{{"generated_sentence": <生成的句子>, "similarity": {similarity}}}'
            }
        ]

        try:
            response = vllm.generate(prompt)
            response = json.loads(response.text)[
                'choices'][0]['message']['content']

            if '```json' in response:
                response = response.split('```json')[-1].split('```')[0]

            result = json.loads(response)
            generated_sentence = result.get("generated_sentence", "").strip()
            similarity = result.get("similarity", 0.0)
            generated_pairs.append({
                "sentence1": original_sentence,
                "sentence2": generated_sentence,
                "score": similarity
            })
        except Exception as e:
            print(
                f"生成失败，句子: {original_sentence[:30]}... 错误: {e} response: {response}")
            continue
    return generated_pairs


STRATEGIES = {
    "llm_search": llm_search_strategy,
    "mask_and_complete": mask_and_complete_strategy,
}


def generate_datasets(args, vllm):
    generate_strategy = args.generate_strategy
    dataset_path = args.dataset_path
    target_type = args.target_type
    input_key = args.input_key
    load_generated = args.load_generated
    save_generated = args.save_generated
    save_path = args.save_path

    if load_generated and save_path and os.path.exists(save_path):
        print(f"加载已生成的数据集: {save_path}")
        with open(save_path, "r", encoding="utf-8") as f:
            generated_pairs = json.load(f)
        return Dataset.from_list(generated_pairs)

    if dataset_path.endswith('offline_test_dataset'):
        dataset = load_jsonl_dataset(dataset_path)
    else:
        dataset = load_dataset(dataset_path)

    dataset = sample_dataset(dataset)

    generated_pairs = []
    if target_type == "a_b_score":
        similarity_scores = [0.1, 0.2, 0.3,
                             0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        for item in dataset:
            original_sentence = item.get(input_key, "").strip()
            if not original_sentence:
                continue

            strategy_func = STRATEGIES.get(generate_strategy)
            if not strategy_func:
                raise ValueError(f"未知策略: {generate_strategy}")

            results = strategy_func(original_sentence, similarity_scores, vllm)
            generated_pairs.extend(results)

    new_dataset = Dataset.from_list(generated_pairs)

    if save_generated and save_path:
        print(f"保存生成的数据集到: {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(generated_pairs, f, ensure_ascii=False, indent=4)

    return new_dataset
