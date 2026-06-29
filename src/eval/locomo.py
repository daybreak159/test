"""
LoCoMo dataset evaluator.

Handles evaluation for LoCoMo dataset with its specific QA format and categories.
"""
import hashlib
import math
import random
from collections import defaultdict
from typing import List, Dict, Any, Tuple

from .base import Evaluator, EvalResult, register_evaluator


@register_evaluator("locomo")
class LoCoMoEvaluator(Evaluator):
    """
    Evaluator for LoCoMo dataset.

    LoCoMo has 5 question categories:
    - 1: Multi-hop
    - 2: Temporal
    - 3: Open-domain
    - 4: Single-hop
    - 5: Adversarial
    """

    # Categories to skip during evaluation
    SKIP_CATEGORIES = {5}  # Skip adversarial questions

    def prepare_eval_args(self) -> Any:
        """Override default eval args to allow longer generations for HotpotQA."""
        eval_args = super().prepare_eval_args()
        eval_args.max_new_tokens = 32
        return eval_args

    def filter_qa_list(self, qa_list: List[Dict]) -> List[Tuple[int, Dict]]:
        """
        Filter QA list, skipping adversarial questions (category 5).

        Args:
            qa_list: List of QA dicts

        Returns:
            List of (index, qa_dict) tuples for valid QA items
        """
        valid_qa = []
        for i, qa in enumerate(qa_list):
            try:
                category = int(qa.get('category', 1))
            except (TypeError, ValueError):
                category = 1
            if category not in self.SKIP_CATEGORIES:
                valid_qa.append((i, qa))
        return valid_qa

    def _get_train_sampling_ratio(self) -> float:
        ratio = getattr(self.args, "locomo_train_query_sampling_ratio", 1.0)
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            ratio = 1.0
        if ratio <= 0.0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        return ratio

    def _build_sampling_rng(self,
                            conversation_id: str = None,
                            outer_epoch: int = 0,
                            inner_epoch: int = 0) -> random.Random:
        base_seed = getattr(self.args, "seed", 42)
        payload = f"{base_seed}|{conversation_id or ''}|{int(outer_epoch)}|{int(inner_epoch)}"
        seed_value = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)
        return random.Random(seed_value)

    def _allocate_category_quotas(self,
                                  bucket_sizes: Dict[int, int],
                                  target_total: int) -> Dict[int, int]:
        total_items = sum(bucket_sizes.values())
        categories = sorted(bucket_sizes.keys())
        if total_items <= 0 or target_total <= 0 or not categories:
            return {}

        quotas = {category: 0 for category in categories}
        effective_target_total = max(target_total, len(categories))
        for category in categories:
            quotas[category] = 1

        remaining = effective_target_total - sum(quotas.values())
        if remaining <= 0:
            return quotas

        expected_additional = {}
        for category in categories:
            expected = remaining * (bucket_sizes[category] / total_items)
            expected_additional[category] = expected

        remainders = []
        for category in categories:
            capacity = max(0, bucket_sizes[category] - quotas[category])
            floor_value = min(capacity, int(math.floor(expected_additional[category])))
            quotas[category] += floor_value
            fractional = expected_additional[category] - math.floor(expected_additional[category])
            remainders.append((fractional, bucket_sizes[category], category))

        remaining = effective_target_total - sum(quotas.values())
        if remaining <= 0:
            return quotas

        for _, _, category in sorted(remainders, key=lambda item: (-item[0], -item[1], item[2])):
            if remaining <= 0:
                break
            if quotas[category] >= bucket_sizes[category]:
                continue
            quotas[category] += 1
            remaining -= 1

        if remaining > 0:
            for category in sorted(categories, key=lambda item: (-bucket_sizes[item], item)):
                while remaining > 0 and quotas[category] < bucket_sizes[category]:
                    quotas[category] += 1
                    remaining -= 1
                if remaining <= 0:
                    break

        return quotas

    def sample_train_qa_list(self,
                             qa_list: List[Dict],
                             valid_qa: List[Tuple[int, Dict]],
                             conversation_id: str = None,
                             outer_epoch: int = 0,
                             inner_epoch: int = 0) -> List[Tuple[int, Dict]]:
        ratio = self._get_train_sampling_ratio()
        if ratio >= 1.0 or len(valid_qa) <= 1:
            return valid_qa

        total_valid = len(valid_qa)
        target_total = max(1, min(total_valid, int(math.ceil(total_valid * ratio))))
        if target_total >= total_valid:
            return valid_qa

        buckets = defaultdict(list)
        for qa_idx, qa in valid_qa:
            try:
                category = int(qa.get("category", 1))
            except (TypeError, ValueError):
                category = 1
            buckets[category].append((qa_idx, qa))

        quotas = self._allocate_category_quotas(
            bucket_sizes={category: len(items) for category, items in buckets.items()},
            target_total=target_total
        )
        rng = self._build_sampling_rng(
            conversation_id=conversation_id,
            outer_epoch=outer_epoch,
            inner_epoch=inner_epoch
        )

        sampled: List[Tuple[int, Dict]] = []
        for category in sorted(buckets.keys()):
            bucket = list(buckets[category])
            quota = min(len(bucket), quotas.get(category, 0))
            if quota <= 0:
                continue
            if quota >= len(bucket):
                sampled.extend(bucket)
            else:
                sampled.extend(rng.sample(bucket, quota))

        sampled.sort(key=lambda item: item[0])
        return sampled

    def build_prompt(self, question: str, retrieved_memories: List[str],
                     qa_item: Dict) -> str:
        """
        Build evaluation prompt for LoCoMo.

        Args:
            question: The question to answer
            retrieved_memories: List of retrieved memory texts
            qa_item: Original QA dict

        Returns:
            Formatted prompt string
        """
        from prompts.prompt_pool import QA_PROMPT

        if len(retrieved_memories) > 0:
            context = "Below is relevant information from the conversation history:\n\n"
            context += "\n\n".join(retrieved_memories)
        else:
            context = "No relevant information available."

        return context + "\n\n" + QA_PROMPT.format(question)

    def get_ground_truth(self, qa_item: Dict) -> str:
        """
        Extract ground truth answer.

        For open-domain (category 3), use only the first answer (split by ';').

        Args:
            qa_item: QA dict

        Returns:
            Ground truth answer string
        """
        answer = str(qa_item.get('answer', ''))
        category = qa_item.get('category', 1)

        # For open-domain, use only first answer
        if category == 3:
            answer = answer.split(';')[0].strip()

        return answer

    def compute_f1(self, prediction: str, ground_truth: str, qa_item: Dict = None) -> float:
        """
        Compute F1 score based on question category.

        - Category 1 (Multi-hop): Use f1_max (handles comma-separated sub-answers)
        - Category 2, 3, 4: Use standard f1_score

        Args:
            prediction: Model prediction
            ground_truth: Ground truth answer
            qa_item: Original QA dict (needed for category info)

        Returns:
            F1 score (0-1)
        """
        from eval_utils import f1_score, f1_max
        if qa_item is None:
            return f1_score(prediction, ground_truth)

        category = qa_item.get('category', 1)

        # Multi-hop: use f1_max for comma-separated sub-answers
        if category == 1:
            return f1_max(prediction, ground_truth)
        else:
            return f1_score(prediction, ground_truth)

    def _get_result_metadata(self, qa: Dict) -> Dict[str, Any]:
        """
        Extract LoCoMo-specific metadata.

        Args:
            qa: QA dict

        Returns:
            Metadata including category and evidence
        """
        return {
            'category': qa.get('category', 1),
            'evidence': qa.get('evidence', [])
        }

    def compute_category_scores(self, results: List[EvalResult]) -> Dict[int, Dict[str, float]]:
        """
        Compute scores grouped by category.

        Args:
            results: List of EvalResult objects

        Returns:
            Dict mapping category to score statistics
        """
        category_results = {}

        for result in results:
            category = result.metadata.get('category', 1)
            if category not in category_results:
                category_results[category] = {
                    'f1_scores': [],
                    'llm_judge_scores': [],
                    'count': 0
                }

            category_results[category]['f1_scores'].append(result.f1_score)
            category_results[category]['llm_judge_scores'].append(result.llm_judge_score)
            category_results[category]['count'] += 1

        # Compute averages
        category_scores = {}
        for category, data in category_results.items():
            import numpy as np
            category_scores[category] = {
                'avg_f1': np.mean(data['f1_scores']) if data['f1_scores'] else 0.0,
                'avg_llm_judge': np.mean(data['llm_judge_scores']) if data['llm_judge_scores'] else 0.0,
                'count': data['count']
            }

        return category_scores
