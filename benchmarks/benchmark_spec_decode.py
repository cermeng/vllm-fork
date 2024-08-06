"""Benchmark the latency of processing a single batch of requests."""
import argparse
import json
import time
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          PreTrainedTokenizerBase)

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptInputs
from vllm.outputs import RequestOutput
from vllm.sequence import RequestMetrics
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
from vllm.utils import FlexibleArgumentParser

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Shuffle the dataset.
    random.shuffle(dataset)

    # Filter out sequences that are too long or too short
    filtered_dataset: List[Tuple[str, int, int]] = []
    for i in range(len(dataset)):
        if len(filtered_dataset) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt = dataset[i][0]
        prompt_token_ids = tokenizer(prompt).input_ids
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = len(completion_token_ids
                         ) if fixed_output_len is None else fixed_output_len
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    return filtered_dataset

def main(args: argparse.Namespace):
    print(args)

    # modify args
    llm = LLM(
        model=args.model,
        speculative_model=args.speculative_model,
        num_speculative_tokens=args.num_speculative_tokens,
        speculative_draft_tensor_parallel_size=\
            args.speculative_draft_tensor_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        ray_workers_use_nsight=args.ray_workers_use_nsight,
        use_v2_block_manager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # (TODO): add args for sampling
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.output_len,
    )
    print(sampling_params)
    tokenizer = llm.get_tokenizer()
    requests = sample_requests(args.dataset_path, args.num_prompts,
                                tokenizer, args.output_len)
    inputs: List[PromptInputs] = [prompt for prompt, _, _ in requests]

    def run_to_completion_v2():
        start_time = time.perf_counter()
        outputs = llm.generate(inputs,
                     sampling_params=sampling_params,
                     use_tqdm=False)
        end_time = time.perf_counter()
        e2e_latency = end_time - start_time
        return e2e_latency, outputs
    
    def get_spec_metrics(outputs: List[RequestOutput]):
        for output in outputs:
            print(output.metrics)

    # print("Warming up...")
    # for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
    #     run_to_completion_v2()
    
    e2e_latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        e2e_latency, outputs = run_to_completion_v2()
        e2e_latencies.append(e2e_latency)
        get_spec_metrics(outputs)
    print(e2e_latencies)
    import pdb
    pdb.set_trace()
    # def run_to_completion(profile_dir: Optional[str] = None):
    #     if profile_dir:
    #         with torch.profiler.profile(
    #                 activities=[
    #                     torch.profiler.ProfilerActivity.CPU,
    #                     torch.profiler.ProfilerActivity.CUDA,
    #                 ],
    #                 on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #                     str(profile_dir))) as p:
    #             llm.generate(dummy_inputs,
    #                          sampling_params=sampling_params,
    #                          use_tqdm=False)
    #         print(p.key_averages())
    #     else:
    #         start_time = time.perf_counter()
    #         llm.generate(dummy_inputs,
    #                      sampling_params=sampling_params,
    #                      use_tqdm=False)
    #         end_time = time.perf_counter()
    #         latency = end_time - start_time
    #         return latency

    # print("Warming up...")
    # for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
    #     run_to_completion(profile_dir=None)

    # if args.profile:
    #     profile_dir = args.profile_result_dir
    #     if not profile_dir:
    #         profile_dir = Path(
    #             "."
    #         ) / "vllm_benchmark_result" / f"latency_result_{time.time()}"
    #     print(f"Profiling (results will be saved to '{profile_dir}')...")
    #     run_to_completion(profile_dir=profile_dir)
    #     return

    # # Benchmark.
    # latencies = []
    # for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
    #     latencies.append(run_to_completion(profile_dir=None))
    # latencies = np.array(latencies)
    # percentages = [10, 25, 50, 75, 90, 99]
    # percentiles = np.percentile(latencies, percentages)
    # print(f'Avg latency: {np.mean(latencies)} seconds')
    # for percentage, percentile in zip(percentages, percentiles):
    #     print(f'{percentage}% percentile latency: {percentile} seconds')

    # # Output JSON results if specified
    # if args.output_json:
    #     results = {
    #         "avg_latency": np.mean(latencies),
    #         "latencies": latencies.tolist(),
    #         "percentiles": dict(zip(percentages, percentiles.tolist())),
    #     }
    #     with open(args.output_json, "w") as f:
    #         json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--speculative-model', type=str, default=None)
    parser.add_argument('--num-speculative-tokens', type=int, default=None)
    parser.add_argument('--speculative-draft-tensor-parallel-size',
                        '-spec-draft-tp',
                        type=int,
                        default=None)
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=10,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=30,
                        help='Number of iterations to run.')
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
 
    parser.add_argument("--enable-prefix-caching",
                        action='store_true',
                        help="Enable automatic prefix caching")
    parser.add_argument(
        "--ray-workers-use-nsight",
        action='store_true',
        help="If specified, use nsight to profile ray workers",
    )
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.9,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    parser.add_argument(
        '--distributed-executor-backend',
        choices=['ray', 'mp'],
        default=None,
        help='Backend to use for distributed serving. When more than 1 GPU '
        'is used, will be automatically set to "ray" if installed '
        'or "mp" (multiprocessing) otherwise.')
    parser.add_argument('--dataset-path', type=str, default=None, help='Path to the dataset.')
    parser.add_argument("--output-len",
                        type=int,
                        default=None,
                        help="Output length for each request. Overrides the "
                        "output length from the dataset.")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=5,
                        help="Number of prompts to process.")
    
    # add sampling param
    parser.add_argument("--temperaturasda2",
                        type=float,
                        default=None,
                        help=".")
    args = parser.parse_args()
    main(args)
