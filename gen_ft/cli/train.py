import argparse

from gen_ft.dataset.dataset_generator import generate_datasets
from gen_ft.trainer.sentence_transformer_trainer import sentence_transformer_train
from gen_ft.utils.vllm_engine import create_vllm_engine, create_vllm_engine_remote


def train(args):
    vllm = None
    if args.vllm_model_path:
        print(
            f"Start to init vllm engine with path = {args.vllm_model_path}...")
        vllm = create_vllm_engine(args)
    else:
        vllm = create_vllm_engine_remote(args)

    print(
        f"Start to generate datasets with input dataset = {args.dataset_path}, strategy = {args.generate_strategy}")
    dataset = generate_datasets(args, vllm)

    print(f"Start to train with sentence_transformer")
    sentence_transformer_train(args, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SentenceTransformer
    parser.add_argument("--pretrain_model_path", type=str,
                        default="all-MiniLM-L6-v2")
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--run_name", type=str, default="GenFT")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)

    # vLLM
    parser.add_argument("--vllm_model_path", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    # API
    parser.add_argument("--api_model", type=str, default="gpt-4o")
    parser.add_argument("--api_url", type=str,
                        default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--api_key", type=str, default="s")
    # Common
    parser.add_argument("--max_model_len", type=int, default=1024)

    # Dataset
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--generate_strategy", type=str, default="llm_search")
    parser.add_argument("--target_type", type=str, default="a_b_score")
    parser.add_argument("--input_key", type=str, default="doc")
    parser.add_argument("--load_generated", action="store_true")
    parser.add_argument("--save_generated", action="store_true")
    parser.add_argument("--save_path", type=str, default=None)

    args = parser.parse_args()

    train(args)
