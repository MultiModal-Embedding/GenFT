from vllm import LLM, SamplingParams


def create_vllm_engine(args):
    try:
        engine = LLM(
            model=args.vllm_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len
        )
        return engine
    except Exception as e:
        print(f"Error loading vLLM model: {e}")
        return None
