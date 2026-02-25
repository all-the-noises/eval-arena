BENCHMARK_DATASET_URLS = {
    # CRUXEval
    "CRUXEval-input": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    "CRUXEval-output": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    "CRUXEval-input-T0.2": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    "CRUXEval-input-T0.8": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    "CRUXEval-output-T0.2": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    "CRUXEval-output-T0.8": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    "cruxeval_input_cot": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    "cruxeval_output_cot": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#cruxeval",
    # HumanEval / plus
    "humaneval": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#humaneval--plus",
    "human_eval": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#humaneval--plus",
    "humaneval+": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#humaneval--plus",
    "human_eval_plus": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#humaneval--plus",
    # MBPP /MBPP+
    "mbpp": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#mbpp--mbpp",
    "mbpp+": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#mbpp--mbpp",
    # LiveCodeBench
    "lcb_codegen_v5": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#livecodebench-lcb_codegen_v5-v6-v6_080124",
    "lcb_codegen_v6": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#livecodebench-lcb_codegen_v5-v6-v6_080124",
    "lcb_codegen_v6_080124": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#livecodebench-lcb_codegen_v5-v6-v6_080124",
    # DS-1000
    "DS1000": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#ds-1000",
    "ds1000": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#ds-1000",
    # SWE-bench
    "swebench-lite": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-verified": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-test": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-bash-only": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-multimodal": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-pro": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#swe-bench-test-lite-verified-bash-only-multimodal",
    # Terminal-Bench
    "terminal-bench-1.0": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#terminal-bench-10-20",
    "terminal-bench-2.0": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#terminal-bench-10-20",
    # SAFIM
    "safim": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#safim",
    # ARC Challenge
    "arc_challenge": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#arc-challenge",
    # HellaSwag
    "hellaswag": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#hellaswag",
    # MMLU
    "mmlu": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#mmlu",
    "mmlu_pro_cot": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#mmlu",
    # PIQA
    "piqa": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#piqa",
    # Social IQA (siqa)
    "siqa": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#social-iqa-siqa",
    # Natural Questions (nq)
    "nq": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#natural-questions-nq",
    # TriviaWA (tqa)
    "tqa": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#triviaqa-tqa",
    # GSM8K
    "gsm8k": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#gsm8k",
    "gsm8k_cot": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#gsm8k",
    "gsm8k_plus_cot": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#gsm8k",
    "mgsm_cot": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#gsm8k",
    # AGIEval (English subset, agi_english)
    "agi_english": "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md#agieval-english-subset-agi_english",
    # AGIEval subsets
    "ap_cot": "https://huggingface.co/datasets/baber/agieval",
    "gmat_cot": "https://huggingface.co/datasets/baber/agieval",
    "lsat_cot": "https://huggingface.co/datasets/baber/agieval",
    "gre_physics_cot": "https://huggingface.co/datasets/baber/agieval",
    # MATH
    "math500_cot": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
    "math_cot": "https://huggingface.co/datasets/hendrycks/competition_math",
    # GPQA
    "gpqa_cot": "https://huggingface.co/datasets/Idavidrein/gpqa",
    # BIG-Bench Hard
    "bbh_cot": "https://huggingface.co/datasets/lukaemon/bbh",
    # LeetCode
    "leetcode": "https://huggingface.co/datasets/greengerong/leetcode",
    # JEEBench
    "jeebench_chat_cot": "https://huggingface.co/datasets/daman1209arora/jeebench",
    # AIME
    "aime2024_cot": "https://huggingface.co/datasets/AI-MO/aimo-validation-aime",
    "aime2025_cot": "https://huggingface.co/datasets/AI-MO/aimo-validation-aime",
}
