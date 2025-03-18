import json

from datasets import load_dataset


def generate_datasets(generate_strategy, dataset_path, vllm, target_type, input_key):
    dataset = load_dataset(dataset_path)
    generated_pairs = []
    if generate_strategy == "llm_search":
        if target_type == "a_b_score":
            similarity_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
            for item in dataset:
                original_sentence = item.get(input_key, "").strip()
                if not original_sentence:
                    continue

                for score in similarity_scores:
                    prompt = (
                        f"请为下面的句子生成一个同义句，使得生成句与原句的语义相似度大约为 {score}。\n"
                        f"原句：{original_sentence}\n"
                        f"请以 JSON 格式返回结果，格式为："
                        f'{{"generated_sentence": <同义句>, "similarity": {score}}}'
                    )
                    try:
                        # 使用 vLLM 生成结果
                        response = vllm.generate(prompt)
                        result = json.loads(response)
                        generated_sentence = result.get("generated_sentence", "").strip()
                        similarity = result.get("similarity", score)

                        generated_pairs.append({
                            "sentence_A": original_sentence,
                            "sentence_B": generated_sentence,
                            "similarity_score": similarity
                        })
                    except Exception as e:
                        print(f"生成失败，句子: {original_sentence[:30]}... 错误: {e}")
                        continue
