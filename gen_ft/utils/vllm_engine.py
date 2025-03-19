import requests
from vllm import LLM, SamplingParams


class LLMEngine:
    def __init__(self, llm, api_url, api_key, model, args):
        self.llm = llm
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.args = args
        if not self.llm:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

    def generate(self, prompt):
        if self.llm:
            return self.llm.generate(prompt)
        else:
            return self._generate_remote(prompt)

    def _generate_remote(self, prompt):
        payload = {
            "model": self.model,
            "messages": prompt,
            "max_tokens": self.args.max_model_len
        }
        return requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            verify=False
        )


def create_vllm_engine(args):
    try:
        engine = LLM(
            model=args.vllm_model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len
        )
        return LLMEngine(engine, "", "", "", args)
    except Exception as e:
        print(f"Error loading vLLM model: {e}")
        return None


def create_vllm_engine_remote(args):
    return LLMEngine(None, args.api_url, args.api_key, args.api_model, args)
