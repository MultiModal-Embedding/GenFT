import argparse

from gen_ft.dataset.dataset_generator import generate_datasets
from gen_ft.trainer.sentence_transformer_trainer import sentence_transformer_train
from gen_ft.utils.vllm_engine import create_vllm_engine


def train(args):
    vllm = None
    if args.vllm_model_path:
        print(f"Start to init vllm engine with path = {args.vllm_model_path}...")
        vllm = create_vllm_engine(args.vllm_model_path)

    print(f"Start to generate datasets with input dataset = {args.dataset_path}, strategy = {args.generate_strategy}")
    dataset = generate_datasets(args.generate_strategy, args.dataset_path, vllm)

    print(f"Start to train with sentence_transformer")
    sentence_transformer_train(args, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SentenceTransformer
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--run_name", type=str, default="GenFT")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--bf16", type=bool, default=False)
    parser.add_argument("--batch_sampler", type=str, default="BatchSamplers.NO_DUPLICATES")
    parser.add_argument("--eval_strategy", type=str, default="steps")
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=100)

    # vLLM
    parser.add_argument("--vllm_model_path", type=str, default=None)

    # Dataset
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--generate_strategy", type=str, default="llm_search")

    args = parser.parse_args()

    train(args)
