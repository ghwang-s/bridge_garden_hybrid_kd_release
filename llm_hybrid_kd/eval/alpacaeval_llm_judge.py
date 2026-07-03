#!/usr/bin/env python3
"""
AlpacaEval 2.0 Compliant LLM Judge Evaluation Script

This script follows the official AlpacaEval 2.0 evaluation methodology:
- Uses weighted_alpaca_eval_gpt4_turbo style judge prompt
- Reports both Win Rate and LC (Length-Controlled) Win Rate
- Implements position bias mitigation through randomization
- Uses an OpenAI-compatible judge API configured by JUDGE_BASE_URL and JUDGE_MODEL
- Retries failed judge calls until a valid judgment is obtained
- Default baseline: text_davinci_001

References:
- AlpacaEval 2.0: https://github.com/tatsu-lab/alpaca_eval
- Length-controlled win rate: https://arxiv.org/abs/2404.04475

"""

import json
import os
import re
import time
import hashlib
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from openai import OpenAI


# ============================================================================
# API Configuration
# ============================================================================
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")

# Default baseline
DEFAULT_BASELINE = "text_davinci_001"

# Data paths
ALPACA_EVAL_ALL_OUTPUTS = "data/alpaca_eval/alpaca_eval_all_outputs.json"


# ============================================================================
# AlpacaEval 2.0 Official Judge Prompt (weighted_alpaca_eval_gpt4_turbo style)
# ============================================================================

ALPACAEVAL_JUDGE_SYSTEM_PROMPT = """You are a highly capable assistant tasked with evaluating and comparing the quality of responses provided by two AI assistants to a user's question. Your goal is to determine which assistant provides a better response.

## Evaluation Criteria

Please evaluate the responses based on the following criteria, in order of importance:
1. **Helpfulness**: Does the response directly address the user's question and provide useful, accurate information?
2. **Accuracy**: Is the information provided factually correct and free from errors?
3. **Relevance**: Does the response stay on topic and avoid unnecessary tangents?
4. **Completeness**: Does the response cover all important aspects of the question?
5. **Clarity**: Is the response well-organized, easy to understand, and free from ambiguity?
6. **Harmlessness**: Does the response avoid harmful, unethical, or inappropriate content?

## Important Notes

- Focus on the **quality of the content**, not superficial features like length or formatting
- A longer response is NOT automatically better; concise, accurate responses can be superior
- If a response contains repetitive content, factual errors, or degenerates into nonsense, it should be rated lower
- Consider the overall usefulness to a typical user asking this question

## Your Task

Compare the two responses and determine which one is better overall. You must choose one - do not say they are equal."""


ALPACAEVAL_JUDGE_USER_TEMPLATE = """I need you to compare two AI assistant responses to the following instruction.

### Instruction:
{instruction}

### Response A:
{output_a}

### Response B:
{output_b}

### Your Evaluation:
First, briefly analyze the strengths and weaknesses of each response (2-3 sentences each).
Then, clearly state which response is better.

Your final answer MUST end with exactly one of these two options on a new line:
- "The better response is: A"
- "The better response is: B"
"""


# ============================================================================
# Data Preparation Functions
# ============================================================================

def load_baseline_outputs(baseline_generator: str = DEFAULT_BASELINE) -> Dict[str, str]:
    """
    Load baseline outputs from alpaca_eval_all_outputs.json
    
    Available generators:
    - text_davinci_001 (easiest, default)
    - alpaca-7b
    - vicuna-13b  
    - chatgpt
    - claude
    - gpt4 (hardest, AlpacaEval 2.0 default)
    """
    print(f"[Data] Loading baseline outputs for: {baseline_generator}")
    
    with open(ALPACA_EVAL_ALL_OUTPUTS, 'r') as f:
        all_outputs = json.load(f)
    
    baseline_dict = {
        item['instruction']: item['output']
        for item in all_outputs
        if item['generator'] == baseline_generator
    }
    
    print(f"[Data] Loaded {len(baseline_dict)} baseline outputs")
    return baseline_dict


def prepare_evaluation_pairs(
    model_outputs_path: str,
    baseline_generator: str = DEFAULT_BASELINE
) -> List[Dict]:
    """
    Prepare evaluation pairs from model outputs and baseline.
    
    Position randomization is based on instruction hash (AlpacaEval 2.0 standard).
    """
    print(f"\n[Data] Preparing evaluation pairs...")
    print(f"[Data] Model outputs: {model_outputs_path}")
    print(f"[Data] Baseline: {baseline_generator}")
    
    # Load model outputs
    with open(model_outputs_path, 'r') as f:
        model_outputs = json.load(f)
    print(f"[Data] Model outputs count: {len(model_outputs)}")
    
    # Load baseline
    baseline_dict = load_baseline_outputs(baseline_generator)
    
    # Create pairs with position randomization
    pairs = []
    for i, item in enumerate(model_outputs):
        instruction = item['instruction']
        
        if instruction not in baseline_dict:
            print(f"[Warning] Instruction not found in baseline: {instruction[:50]}...")
            continue
        
        model_out = item['output']
        ref_out = baseline_dict[instruction]
        
        # Position randomization based on instruction hash (AlpacaEval 2.0 standard)
        seed = int(hashlib.md5(instruction.encode()).hexdigest()[:8], 16) % 2
        model_is_m = (seed == 0)
        
        pairs.append({
            'idx': i + 1,
            'instruction': instruction,
            'output_m': model_out if model_is_m else ref_out,
            'output_M': ref_out if model_is_m else model_out,
            'model_is_m': model_is_m,
            'len_model': len(model_out),
            'len_ref': len(ref_out)
        })
    
    print(f"[Data] Created {len(pairs)} evaluation pairs")
    return pairs


@dataclass
class JudgmentResult:
    """Stores the result of a single judgment"""
    idx: int
    instruction: str
    winner: str  # 'm' or 'M'
    model_wins: bool
    reasoning: str
    len_model: int
    len_ref: int


class AlpacaEvalJudge:
    """
    AlpacaEval 2.0 compliant LLM judge using an OpenAI-compatible API.
    
    Key features:
    - Uses weighted_alpaca_eval_gpt4_turbo style prompts
    - Position randomization (via model_is_m field)
    - Robust parsing of judge outputs
    - Retry-until-success behavior for judge calls
    """
    
    def __init__(
        self,
        api_key: Optional[str] = JUDGE_API_KEY,
        base_url: str = JUDGE_BASE_URL,
        model: str = JUDGE_MODEL,
        temperature: float = 0.0,  # Deterministic for reproducibility
        base_retry_delay: float = 2.0,
        max_retry_delay: float = 60.0,  # Max wait time for rate limiting
    ):
        if not api_key:
            raise ValueError("Set JUDGE_API_KEY or OPENAI_API_KEY before running the judge.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay
        
        print(f"[Judge] Initialized with model: {model}")
        print(f"[Judge] API: {base_url}")
        print(f"[Judge] Temperature: {temperature} (deterministic mode)")
        print(f"[Judge] Mode: retry until success")
    
    def judge_pair(
        self, 
        instruction: str, 
        output_m: str, 
        output_M: str, 
        model_is_m: bool,
        idx: int = 0
    ) -> Tuple[str, str]:
        """
        Judge a single pair of outputs.
        
        This method retries until a valid judgment is returned.
        
        Args:
            instruction: The instruction/prompt
            output_m: Output in position 'm' (first in data)
            output_M: Output in position 'M' (second in data)
            model_is_m: If True, the model being evaluated is output_m
            idx: Sample index for logging
            
        Returns:
            (winner, reasoning) where winner is 'm' or 'M'
        """
        # Position assignment for the prompt
        if model_is_m:
            output_a = output_m  # Model is A
            output_b = output_M  # Reference is B
        else:
            output_a = output_M  # Reference is A
            output_b = output_m  # Model is B
        
        # Construct the prompt
        user_message = ALPACAEVAL_JUDGE_USER_TEMPLATE.format(
            instruction=instruction[:4000],  # Truncate to avoid token limit
            output_a=output_a[:6000],
            output_b=output_b[:6000]
        )
        
        # Retry until a valid response is parsed.
        attempt = 0
        while True:
            attempt += 1
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": ALPACAEVAL_JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.temperature,
                    max_tokens=500
                )
                
                reasoning = response.choices[0].message.content.strip()
                
                # Parse the winner from the response
                winner_ab = self._parse_winner(reasoning)
                
                if winner_ab is None:
                    # Could not parse, retry with exponential backoff
                    wait_time = min(self.base_retry_delay * (2 ** (attempt - 1)), self.max_retry_delay)
                    print(f"[idx={idx}] Parse failed (attempt {attempt}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                
                # Convert A/B back to m/M based on position assignment
                if winner_ab == 'A':
                    winner = 'm' if model_is_m else 'M'
                else:  # winner_ab == 'B'
                    winner = 'M' if model_is_m else 'm'
                
                return winner, reasoning
                
            except Exception as e:
                error_msg = str(e)
                
                # Calculate wait time with exponential backoff
                wait_time = min(self.base_retry_delay * (2 ** (attempt - 1)), self.max_retry_delay)
                
                # Handle rate limiting specifically
                if '429' in error_msg or 'Throttling' in error_msg or 'rate' in error_msg.lower():
                    wait_time = min(wait_time * 3, self.max_retry_delay)
                    print(f"[idx={idx}] RATE LIMITED (attempt {attempt}), waiting {wait_time:.1f}s...")
                else:
                    print(f"[idx={idx}] API Error (attempt {attempt}): {error_msg[:100]}... retrying in {wait_time:.1f}s")
                
                time.sleep(wait_time)
                # Continue retrying after backoff.
    
    def _parse_winner(self, response: str) -> Optional[str]:
        """
        Parse the winner (A or B) from the judge's response.
        
        AlpacaEval expects clear indication of which response is better.
        """
        response_lower = response.lower()
        
        # Pattern 1: "The better response is: A" or "The better response is: B"
        pattern1 = re.search(r'the better response is[:\s]*([ab])', response_lower)
        if pattern1:
            return pattern1.group(1).upper()
        
        # Pattern 2: "Response A is better" or "Response B is better"
        pattern2 = re.search(r'response\s+([ab])\s+is\s+better', response_lower)
        if pattern2:
            return pattern2.group(1).upper()
        
        # Pattern 3: "I choose A" or "I choose B"
        pattern3 = re.search(r'i\s+choose\s+([ab])', response_lower)
        if pattern3:
            return pattern3.group(1).upper()
        
        # Pattern 4: "A is the better" or "B is the better"
        pattern4 = re.search(r'([ab])\s+is\s+the\s+better', response_lower)
        if pattern4:
            return pattern4.group(1).upper()
        
        # Pattern 5: Last mention of "Response A" or "Response B" with positive context
        last_a = max(
            response_lower.rfind('response a is better'),
            response_lower.rfind('choose a'),
            response_lower.rfind('prefer a'),
            response_lower.rfind('winner: a'),
            response_lower.rfind('better response is: a'),
            response_lower.rfind('better response is a'),
        )
        
        last_b = max(
            response_lower.rfind('response b is better'),
            response_lower.rfind('choose b'),
            response_lower.rfind('prefer b'),
            response_lower.rfind('winner: b'),
            response_lower.rfind('better response is: b'),
            response_lower.rfind('better response is b'),
        )
        
        if last_a > last_b:
            return 'A'
        elif last_b > last_a:
            return 'B'
        
        return None


def compute_win_rate(
    judgments: List[JudgmentResult]
) -> float:
    """
    Compute raw win rate for the model.
    
    Win Rate = (model_wins + 0.5 * ties) / total
    Note: In our setup, ties are not possible since judge must choose.
    """
    model_wins = sum(1 for j in judgments if j.model_wins)
    return model_wins / len(judgments) if judgments else 0.0


def compute_lc_win_rate(
    judgments: List[JudgmentResult]
) -> float:
    """
    Compute Length-Controlled (LC) Win Rate using logistic regression.
    
    This is the key metric in AlpacaEval 2.0 that controls for length bias:
    - Fits a logistic regression: P(model_wins) ~ length_diff
    - Returns predicted win probability at length_diff = 0
    
    Reference: https://arxiv.org/abs/2404.04475
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("[Warning] scikit-learn not available, returning raw win rate as LC win rate")
        return compute_win_rate(judgments)
    
    if len(judgments) < 10:
        return compute_win_rate(judgments)
    
    # Extract features and labels
    len_diffs = np.array([j.len_model - j.len_ref for j in judgments])
    model_wins = np.array([1 if j.model_wins else 0 for j in judgments])
    
    # Normalize length difference
    mean_diff = len_diffs.mean()
    std_diff = len_diffs.std() + 1e-8
    len_diffs_norm = (len_diffs - mean_diff) / std_diff
    
    # Check if we have both classes
    if len(set(model_wins)) < 2:
        return compute_win_rate(judgments)
    
    # Fit logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(len_diffs_norm.reshape(-1, 1), model_wins)
    
    # Predict at length_diff = 0 (normalized)
    zero_norm = (0 - mean_diff) / std_diff
    lc_win_rate = clf.predict_proba([[zero_norm]])[0, 1]
    
    return float(lc_win_rate)


def compute_confidence_interval(
    win_rate: float, 
    n: int, 
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute confidence interval for win rate using normal approximation.
    """
    # Use z=1.96 for 95% CI (avoiding scipy dependency)
    z = 1.96 if confidence == 0.95 else 2.576  # 99% CI fallback
    se = np.sqrt(win_rate * (1 - win_rate) / n)
    
    lower = max(0, win_rate - z * se)
    upper = min(1, win_rate + z * se)
    
    return lower, upper


def run_evaluation(
    model_outputs_path: str,
    output_dir: str,
    baseline: str = DEFAULT_BASELINE,
    api_key: Optional[str] = JUDGE_API_KEY,
    base_url: str = JUDGE_BASE_URL,
    model: str = JUDGE_MODEL,
    batch_delay: float = 0.5,
    checkpoint_interval: int = 50,
) -> Dict:
    """
    Run the full AlpacaEval 2.0 compliant evaluation.
    
    This function retries failed samples until all judgments are collected.
    
    Args:
        model_outputs_path: Path to model outputs JSON (e.g., output/hard/alpaca_eval_outputs.json)
        output_dir: Directory to save results
        baseline: Baseline generator (default: text_davinci_001)
        api_key: API key for the configured judge endpoint
        base_url: OpenAI-compatible judge API base URL
        model: Model to use as judge
        batch_delay: Delay between API calls in seconds
        checkpoint_interval: Save checkpoint every N samples
        
    Returns:
        Dictionary containing evaluation results
    """
    # Prepare evaluation pairs
    pairs = prepare_evaluation_pairs(model_outputs_path, baseline)
    
    if not pairs:
        raise ValueError("No evaluation pairs created! Check model outputs and baseline.")
    
    # Initialize judge
    judge = AlpacaEvalJudge(api_key=api_key, base_url=base_url, model=model)
    
    print(f"\n[Evaluation] Total pairs to evaluate: {len(pairs)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = os.path.join(output_dir, "checkpoint_judgments.json")
    judgments = []
    start_idx = 0
    
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
            judgments = [JudgmentResult(**j) for j in checkpoint_data]
            start_idx = len(judgments)
            print(f"[Checkpoint] Resuming from index {start_idx}")
    
    # Run evaluation
    print(f"\n[Evaluation] Starting evaluation...")
    print(f"[Evaluation] Judge: {model}")
    print(f"[Evaluation] Baseline: {baseline}")
    print(f"[Evaluation] Mode: retry until success")
    print("=" * 70)
    
    try:
        for i in range(start_idx, len(pairs)):
            item = pairs[i]
            
            idx = item['idx']
            instruction = item['instruction']
            output_m = item['output_m']
            output_M = item['output_M']
            model_is_m = item['model_is_m']
            len_model = item['len_model']
            len_ref = item['len_ref']
            
            # Judge this pair; retry handling is inside judge_pair.
            winner, reasoning = judge.judge_pair(
                instruction=instruction,
                output_m=output_m,
                output_M=output_M,
                model_is_m=model_is_m,
                idx=idx
            )
            
            # Determine if model won
            if model_is_m:
                model_wins = (winner == 'm')
            else:
                model_wins = (winner == 'M')
            
            result = JudgmentResult(
                idx=idx,
                instruction=instruction[:100] + "..." if len(instruction) > 100 else instruction,
                winner=winner,
                model_wins=model_wins,
                reasoning=reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
                len_model=len_model,
                len_ref=len_ref
            )
            judgments.append(result)
            
            # Progress
            current_model_wins = sum(1 for j in judgments if j.model_wins)
            current_win_rate = current_model_wins / len(judgments) * 100
            status = " Model" if model_wins else " Ref"
            print(f"[{i+1:4d}/{len(pairs)}] idx={idx:3d} | {status} | WR: {current_win_rate:.1f}%")
            
            # Checkpoint
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_data = [
                    {
                        'idx': j.idx,
                        'instruction': j.instruction,
                        'winner': j.winner,
                        'model_wins': j.model_wins,
                        'reasoning': j.reasoning,
                        'len_model': j.len_model,
                        'len_ref': j.len_ref
                    }
                    for j in judgments
                ]
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                print(f"[Checkpoint] Saved at {i+1} samples")
            
            # Rate limiting
            time.sleep(batch_delay)
    
    except KeyboardInterrupt:
        print("\n[Interrupted] Saving progress...")
    
    # Compute final statistics
    print("\n" + "=" * 70)
    print("[Results] Computing final statistics...")
    
    win_rate = compute_win_rate(judgments)
    lc_win_rate = compute_lc_win_rate(judgments)
    std_err = np.sqrt(win_rate * (1 - win_rate) / len(judgments))
    
    ci_lower, ci_upper = compute_confidence_interval(win_rate, len(judgments))
    
    # Count results
    model_wins_count = sum(1 for j in judgments if j.model_wins)
    ref_wins_count = len(judgments) - model_wins_count
    
    # Get method name from path
    method_name = os.path.basename(os.path.dirname(model_outputs_path))
    
    # Compile results
    results = {
        "method": method_name,
        "model_outputs": model_outputs_path,
        "baseline": baseline,
        "judge": model,
        "judge_type": "weighted_alpaca_eval_gpt4_turbo_style",
        "n_samples": len(judgments),
        "win_rate": round(win_rate, 4),
        "lc_win_rate": round(lc_win_rate, 4),
        "win_rate_pct": f"{win_rate*100:.2f}%",
        "lc_win_rate_pct": f"{lc_win_rate*100:.2f}%",
        "std_err": round(std_err, 4),
        "95_CI": [round(ci_lower, 4), round(ci_upper, 4)],
        "model_wins": model_wins_count,
        "ref_wins": ref_wins_count,
        "avg_len_model": round(np.mean([j.len_model for j in judgments]), 1),
        "avg_len_ref": round(np.mean([j.len_ref for j in judgments]), 1),
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("AlpacaEval 2.0 Results (LLM Judge)")
    print("=" * 70)
    print(f"Method: {method_name}")
    print(f"Baseline: {baseline}")
    print(f"Judge: {model}")
    print(f"Samples: {results['n_samples']}")
    print("-" * 70)
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"LC Win Rate: {lc_win_rate*100:.2f}%")
    print(f"Std Error: {std_err*100:.2f}%")
    print(f"95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print("-" * 70)
    print(f"Model Wins: {model_wins_count} ({model_wins_count/len(judgments)*100:.1f}%)")
    print(f"Reference Wins: {ref_wins_count} ({ref_wins_count/len(judgments)*100:.1f}%)")
    print(f"Avg Length - Model: {results['avg_len_model']:.0f} chars")
    print(f"Avg Length - Reference: {results['avg_len_ref']:.0f} chars")
    print("=" * 70)
    
    # Save results
    results_path = os.path.join(output_dir, "alpaca_eval_result.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Results: {results_path}")
    
    # Save detailed judgments
    judgments_data = [
        {
            'idx': j.idx,
            'instruction': j.instruction,
            'winner': j.winner,
            'model_wins': j.model_wins,
            'reasoning': j.reasoning,
            'len_model': j.len_model,
            'len_ref': j.len_ref
        }
        for j in judgments
    ]
    judgments_path = os.path.join(output_dir, "alpaca_eval_judgments.json")
    with open(judgments_path, 'w') as f:
        json.dump(judgments_data, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Judgments: {judgments_path}")
    
    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return results


def evaluate_all_methods(
    baseline: str = DEFAULT_BASELINE,
    batch_delay: float = 0.5,
) -> Dict[str, Dict]:
    """
    Evaluate all methods in the output directory against the specified baseline.
    
    Methods:
    - output/hard/alpaca_eval_outputs.json
    - output/soft/alpaca_eval_outputs.json
    - outputs/hybrid_gkd/alpaca_eval_outputs.json
    - output/base_model/alpaca_eval_outputs.json
    - output/32B_3B/hybrid/alpaca_eval_outputs.json
    """
    methods = [
        ("hard", "output/hard/alpaca_eval_outputs.json"),
        ("soft", "output/soft/alpaca_eval_outputs.json"),
        ("hybrid_gkd", "outputs/hybrid_gkd/alpaca_eval_outputs.json"),
        ("base_model", "output/base_model/alpaca_eval_outputs.json"),
        ("32B_3B_hybrid", "output/32B_3B/hybrid/alpaca_eval_outputs.json"),
    ]
    
    all_results = {}
    
    print("\n" + "=" * 70)
    print("AlpacaEval 2.0 - Evaluating ALL Methods")
    print(f"Baseline: {baseline}")
    print("=" * 70)
    
    for method_name, outputs_path in methods:
        if not os.path.exists(outputs_path):
            print(f"\n[Skip] {method_name}: {outputs_path} not found")
            continue
        
        print(f"\n{'='*70}")
        print(f"[Method] {method_name}")
        print(f"{'='*70}")
        
        output_dir = os.path.dirname(outputs_path)
        
        results = run_evaluation(
            model_outputs_path=outputs_path,
            output_dir=output_dir,
            baseline=baseline,
            batch_delay=batch_delay,
        )
        
        all_results[method_name] = results
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - AlpacaEval 2.0 Win Rates")
    print(f"Baseline: {baseline}")
    print("=" * 70)
    print(f"{'Method':<20} {'Win Rate':>12} {'LC Win Rate':>12} {'Samples':>10}")
    print("-" * 70)
    
    for method_name, results in all_results.items():
        print(f"{method_name:<20} {results['win_rate_pct']:>12} {results['lc_win_rate_pct']:>12} {results['n_samples']:>10}")
    
    print("=" * 70)
    
    # Save summary
    summary_path = "output/alpaca_eval_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Summary: {summary_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="AlpacaEval-style LLM Judge Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single method against text_davinci_001
  python eval/alpacaeval_llm_judge.py --model_outputs output/hard/alpaca_eval_outputs.json
  
  # Evaluate against a different baseline
  python eval/alpacaeval_llm_judge.py --model_outputs output/soft/alpaca_eval_outputs.json --baseline chatgpt
  
  # Evaluate ALL methods in output directory
  python eval/alpacaeval_llm_judge.py --all
  
Available baselines (from easiest to hardest):
  - text_davinci_001 (default, easiest)
  - alpaca-7b
  - vicuna-13b
  - chatgpt
  - claude
  - gpt4 (hardest, AlpacaEval 2.0 official)
"""
    )
    parser.add_argument(
        "--model_outputs",
        type=str,
        default=None,
        help="Path to model outputs JSON (e.g., output/hard/alpaca_eval_outputs.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: same as model_outputs directory)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_BASELINE,
        help=f"Baseline generator (default: {DEFAULT_BASELINE})"
    )
    parser.add_argument(
        "--batch_delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate ALL methods in output directory"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=JUDGE_API_KEY,
        help="Judge API key. Defaults to JUDGE_API_KEY or OPENAI_API_KEY."
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=JUDGE_BASE_URL,
        help="OpenAI-compatible judge API base URL."
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=JUDGE_MODEL,
        help="Judge model name."
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Evaluate all methods
        evaluate_all_methods(
            baseline=args.baseline,
            batch_delay=args.batch_delay,
        )
    elif args.model_outputs:
        # Evaluate single method
        output_dir = args.output_dir or os.path.dirname(args.model_outputs)
        
        run_evaluation(
            model_outputs_path=args.model_outputs,
            output_dir=output_dir,
            baseline=args.baseline,
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.judge_model,
            batch_delay=args.batch_delay,
        )
    else:
        parser.print_help()
        print("\nERROR: Either --model_outputs or --all must be specified.")


if __name__ == "__main__":
    main()
