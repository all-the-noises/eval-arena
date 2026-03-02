docbase = "https://github.com/all-the-noises/eval-arena/blob/main/doc/benchmarks.md"
BENCHMARK_DATASET_URLS = {
    # CRUXEval
    "CRUXEval-input": f"{BASE}#cruxeval",
    "CRUXEval-output": f"{BASE}#cruxeval",
    "CRUXEval-input-T0.2": f"{BASE}#cruxeval",
    "CRUXEval-input-T0.8": f"{BASE}#cruxeval",
    "CRUXEval-output-T0.2": f"{BASE}#cruxeval",
    "CRUXEval-output-T0.8": f"{BASE}#cruxeval",
    "cruxeval_input_cot": f"{BASE}#cruxeval",
    "cruxeval_output_cot": f"{BASE}#cruxeval",
    # HumanEval / plus
    "humaneval": f"{BASE}#humaneval--plus",
    "human_eval": f"{BASE}#humaneval--plus",
    "humaneval+": "#humaneval--plus",
    "human_eval_plus": f"{BASE}#humaneval--plus",
    # MBPP /MBPP+
    "mbpp": f"{BASE}#mbpp--mbpp",
    "mbpp+": f"{BASE}#mbpp--mbpp",
    # LiveCodeBench
    "lcb_codegen_v5": f"{BASE}#livecodebench-lcb_codegen_v5-v6-v6_080124",
    "lcb_codegen_v6": f"{BASE}#livecodebench-lcb_codegen_v5-v6-v6_080124",
    "lcb_codegen_v6_080124": f"{BASE}#livecodebench-lcb_codegen_v5-v6-v6_080124",
    # DS-1000
    "DS1000": f"{BASE}#ds-1000",
    "ds1000": f"{BASE}#ds-1000",
    # SWE-bench
    "swebench-lite": f"{BASE}#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-verified": f"{BASE}#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-test": f"{BASE}#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-bash-only": f"{BASE}#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-multimodal": f"{BASE}#swe-bench-test-lite-verified-bash-only-multimodal",
    "swebench-pro": f"{BASE}#swe-bench-test-lite-verified-bash-only-multimodal",
    # Terminal-Bench
    "terminal-bench-1.0": f"{BASE}#terminal-bench-10-20",
    "terminal-bench-2.0": f"{BASE}#terminal-bench-10-20",
    # SAFIM
    "safim": f"{BASE}#safim",
    # ARC Challenge
    "arc_challenge": f"{BASE}#arc-challenge",
    # HellaSwag
    "hellaswag": f"{BASE}#hellaswag",
    # MMLU
    "mmlu": f"{BASE}#mmlu",
    "mmlu_pro_cot": f"{BASE}#mmlu",
    # PIQA
    "piqa": f"{BASE}#piqa",
    # Social IQA (siqa)
    "siqa": f"{BASE}#social-iqa-siqa",
    # Natural Questions (nq)
    "nq": f"{BASE}#natural-questions-nq",
    # TriviaWA (tqa)
    "tqa": f"{BASE}#triviaqa-tqa",
    # GSM8K
    "gsm8k": f"{BASE}#gsm8k",
    "gsm8k_cot": f"{BASE}#gsm8k",
    "gsm8k_plus_cot": f"{BASE}#gsm8k",
    "mgsm_cot": f"{BASE}#gsm8k",
    # AGIEval (English subset, agi_english)
    "agi_english": f"{BASE}#agieval-english-subset-agi_english",
    # AGIEval subsets
    # "ap_cot": "https://huggingface.co/datasets/baber/agieval",
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
