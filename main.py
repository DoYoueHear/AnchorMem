import sys
sys.path.append(".") 

import os
from typing import List, Optional
import json
import random
import numpy as np
import torch
import dataclasses
from collections import defaultdict
from AnchorMem import AnchorMem
from src.utils.misc_utils import QuerySolution
from src.utils.misc_utils import string_to_bool
from src.utils.config_utils import BaseConfig
from src.evaluation.qa_eval import QAExactMatch, QAF1Score, QABleuScore
from src.datasets.locomo10_loader import (
    make_docs_from_locomo10_conversations,
)
import argparse
import logging


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


SEED = 42
def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_deterministic_seed(SEED)

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('locomo_eval')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception:
                pass

    logger.propagate = False
    return logger


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(self, obj)


def main():
    parser = argparse.ArgumentParser(description="AnchorMem retrieval and QA")
    parser.add_argument('--dataset', type=str, default='locomo', help='Dataset name (used for prompts/config)')
    parser.add_argument('--dataset_path', type=str, default='your/path/to/locomo10.json',
                        help='Path to LoCoMo10 dataset JSON for evaluation')
 
    parser.add_argument('--llm_base_url', type=str, default='http://localhost:8080/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='Qwen2.32B', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='Transformers/all-MiniLM-L6-v2', help='embedding model name')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore existing cached memories and rebuild from scratch.')
    parser.add_argument('--force_fact_extraction_from_scratch', type=str, default='false', help='If set to False, will try to first reuse cached fact results for the corpus if they exist.')
    parser.add_argument('--fact_sim_threshold', type=float, default=0.85, help='Fact similarity threshold for retrieval')
    parser.add_argument('--related_fact_top_k', type=int, default=3, help='Number of related facts to retrieve')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Base save directory')
    args = parser.parse_args()

    dataset_name = args.dataset
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name

    base_save_dir = args.save_dir
    if base_save_dir == 'outputs':
        # test-thing
        base_save_dir = base_save_dir + '/' + dataset_name + '-' + llm_name
    else:
        base_save_dir = base_save_dir + '_' + dataset_name + '-' + llm_name

    os.makedirs(base_save_dir, exist_ok=True)
    
    log_file = os.path.join(base_save_dir, "qa_aggregated.log")
    logger = setup_logger(log_file=log_file)

    with open(args.dataset_path, "r", encoding="utf-8") as f:
        locomo10_samples = json.load(f)

    
    UNANSWERABLE_TEXT = "Not mentioned in the conversation or memory."
    
    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_fact_extraction_from_scratch = string_to_bool(args.force_fact_extraction_from_scratch)

    total_all_recalls = []
    total_categories = []
    total_example_em = []
    total_example_f1 = []
    total_example_bleu = []
    final_metrics_per_category = defaultdict(lambda: {"recalls": [], "ems": [], "f1s": [], "bleus": []})

    for sample_index, sample_data in enumerate(locomo10_samples):
        logger.info(f"\n{'='*30} Processing Sample {sample_index+1} / {len(locomo10_samples)} {'='*30}")

        # sample_index = data_index

        sample_save_dir = os.path.join(base_save_dir, f"sample_{sample_index}")
        os.makedirs(sample_save_dir, exist_ok=True)
        
        current_sample_list = [sample_data]
        docs, docs_with_ids = make_docs_from_locomo10_conversations(
            samples=current_sample_list,
            per_session=True,
            only_referenced=True,
            overlap=1
        )
        logger.info(f"[Sample {sample_index}] Built {len(docs)} docs.")

        id_doc_pairs = []
        for d in docs_with_ids:
            [(doc_id, doc_text)] = d.items()
            id_doc_pairs.append((doc_id, doc_text))

        all_queries: List[str] = []
        all_queries_evidence: List[List[str]] = []
        gold_answers_list: List[List[str]] = []
        categories: List[int] = []

        for qa in sample_data.get("qa", []):
            
            cat = int(qa.get("category"))
            if cat == 5:
                continue

            all_queries.append(qa["question"])
            all_queries_evidence.append(qa["evidence"])
            categories.append(cat)
            if cat == 5:
                gold_answers_list.append([UNANSWERABLE_TEXT])
            else:
                gold_ans = qa.get("answer")
                if isinstance(gold_ans, list):
                    gold_answers_list.append([str(a) for a in gold_ans])
                else:
                    gold_answers_list.append([str(gold_ans)])

        if not all_queries:
            logger.warning(f"[Sample {sample_index}] No QA pairs found. Skipping.")
            continue
            
        config = BaseConfig(
            save_dir=sample_save_dir,
            llm_base_url=llm_base_url,
            llm_name=llm_name,
            dataset=dataset_name,
            embedding_model_name=args.embedding_name,
            force_index_from_scratch=force_index_from_scratch,
            force_fact_extraction_from_scratch=force_fact_extraction_from_scratch,
            rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
            retrieval_top_k=10,
            fact_sim_threshold=args.fact_sim_threshold,
            related_fact_top_k=args.related_fact_top_k,
            linking_top_k=5,
            max_qa_steps=3,
            qa_top_k=5,
            embedding_batch_size=8,
            max_new_tokens=None,
            corpus_len=len(docs),
        )

        logging.basicConfig(level=logging.INFO)

        cache_file_path = os.path.join(sample_save_dir, "queries_solutions.json")
        queries_solutions: Optional[List[QuerySolution]] = None

        if os.path.exists(cache_file_path) and not force_index_from_scratch:
            logger.info(f"[Sample {sample_index}] Loading cached solutions from {cache_file_path}")
            try:
                with open(cache_file_path, 'r', encoding='utf-8') as f:
                    loaded_data = json.load(f)
                
                queries_solutions = []
                for data in loaded_data:
                    if data.get('doc_scores') is not None:
                        data['doc_scores'] = np.array(data['doc_scores'])
                    
                    valid_keys = {f.name for f in dataclasses.fields(QuerySolution)}
                    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                    
                    queries_solutions.append(QuerySolution(**filtered_data))
                logger.info(f"[Sample {sample_index}] Successfully loaded {len(queries_solutions)} cached solutions.")
            except Exception as e:
                logger.warning(f"[Sample {sample_index}] Failed to load cache. Re-running. Error: {e}")
                queries_solutions = None

        if queries_solutions is None:
            logger.info(f"[Sample {sample_index}] No cache found (or forced). Running RAG pipeline...")
            
            anchormem = AnchorMem(global_config=config)
            
            anchormem.index(docs)

            # return
            queries_solutions, _, _ = anchormem.rag_qa(
                queries=all_queries,
                gold_docs=None,
                gold_answers=None,
            )
            
            anchormem.save_cost_summary()

            try:
                with open(cache_file_path, 'w', encoding='utf-8') as f:
                    json.dump(queries_solutions, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)
                logger.info(f"[Sample {sample_index}] Saved {len(queries_solutions)} solutions to cache.")
            except Exception as e:
                logger.warning(f"[Sample {sample_index}] Failed to save cache. Error: {e}")


        
        sample_all_recalls = []
        sample_recalls_by_category = defaultdict(list)

        safe_len = min(len(all_queries), len(queries_solutions), len(categories), len(gold_answers_list))
        
        for i in range(safe_len):
            logger.info(f"[Sample {sample_index}] category: {categories[i]}")
            logger.info(f"[Sample {sample_index}] question: {all_queries[i]}")
            logger.info(f"[Sample {sample_index}] gold_answers: {gold_answers_list[i]}")
            logger.info(f"[Sample {sample_index}] response: {queries_solutions[i].answer}")
            logger.info(f"[Sample {sample_index}] top facts: {queries_solutions[i].topk_facts}")
            logger.info("[Sample {}] top docs: {}".format(sample_index, "\n".join(queries_solutions[i].docs)))

            # retrieved_docs_for_i = queries_solutions[i].docs[:config.qa_top_k]
            retrieved_docs_for_i = queries_solutions[i].docs
            ground_truth_for_i = all_queries_evidence[i]
            ground_truth_set = set(ground_truth_for_i)
            found_relevant_ids = set()

            for retrieved_doc in retrieved_docs_for_i:
                retrieved_doc_norm = retrieved_doc.lower()

                for original_id, original_doc in id_doc_pairs:
                    if original_id not in ground_truth_set:
                        continue
                    if original_doc.lower() in retrieved_doc_norm:
                        found_relevant_ids.add(original_id)

            if len(ground_truth_set) > 0:
                recall_i = len(found_relevant_ids) / len(ground_truth_set)
            else:
                recall_i = 0.0

            sample_all_recalls.append(recall_i)
            
            cat = categories[i]
            sample_recalls_by_category[cat].append(recall_i)
            
            logger.info(f"[Sample {sample_index}] the recall@5={recall_i:.2%}")
            logger.info("")

        logger.info(f"--- [Sample {sample_index}] Recall Metrics ---")
        for cat, recalls in sample_recalls_by_category.items():
            if recalls:
                avg_recall = sum(recalls) / len(recalls)
                logger.info(f"[Sample {sample_index}] Category {cat} Recall: {avg_recall:.2%} (count={len(recalls)})")
        
        if sample_all_recalls:
            macro_average_recall = sum(sample_all_recalls) / len(sample_all_recalls)
            logger.info(f"[Sample {sample_index}] Macro-Average Recall: {macro_average_recall:.2%}")
        logger.info(f"--- [Sample {sample_index}] End Recall Metrics ---")


        predicted_answers: List[str] = [q.answer for q in queries_solutions]

        qa_em_evaluator = QAExactMatch(global_config=config)
        qa_f1_evaluator = QAF1Score(global_config=config)
        qa_bleu_evaluator = QABleuScore(global_config=config)

        _, example_em_list = qa_em_evaluator.calculate_metric_scores(
            gold_answers=gold_answers_list,
            predicted_answers=predicted_answers,
        )
        _, example_f1_list = qa_f1_evaluator.calculate_metric_scores(
            gold_answers=gold_answers_list,
            predicted_answers=predicted_answers,
        )
        _, example_bleu_list = qa_bleu_evaluator.calculate_metric_scores(
            gold_answers=gold_answers_list,
            predicted_answers=predicted_answers,
        )


        total_all_recalls.extend(sample_all_recalls)
        total_categories.extend(categories)
        total_example_em.extend([d["ExactMatch"] for d in example_em_list])
        total_example_f1.extend([d["F1"] for d in example_f1_list])
        total_example_bleu.extend([d["bleu1"] for d in example_bleu_list])
        
        logger.info(f"{'='*30} Finished Sample {sample_index+1} / {len(locomo10_samples)} {'='*30}")


    logger.info(f"\n{'='*80}")
    logger.info(f"All {len(locomo10_samples)} samples processed. Calculating aggregated metrics...")
    logger.info(f"{'='*80}")

    if not total_categories:
        logger.error("No data collected across all samples. Exiting.")
        return

    for i in range(len(total_categories)):
        cat = total_categories[i]
        final_metrics_per_category[cat]["recalls"].append(total_all_recalls[i])
        final_metrics_per_category[cat]["ems"].append(total_example_em[i])
        final_metrics_per_category[cat]["f1s"].append(total_example_f1[i])
        final_metrics_per_category[cat]["bleus"].append(total_example_bleu[i])

    final_per_category_report = {}
    logger.info("--- Aggregated Per-Category Metrics (All Samples) ---")
    
    for cat in sorted(final_metrics_per_category.keys()):
        data = final_metrics_per_category[cat]
        count = len(data["recalls"])
        if count == 0:
            continue
        
        avg_recall = sum(data["recalls"]) / count
        avg_em = sum(data["ems"]) / count
        avg_f1 = sum(data["f1s"]) / count
        avg_bleu = sum(data["bleus"]) / count
        
        final_per_category_report[cat] = {
            "Recall": avg_recall,
            "ExactMatch": avg_em,
            "F1": avg_f1,
            "Bleu1": avg_bleu,
            "count": count
        }
        logger.info(
            f"Category {cat}: Recall={avg_recall:.4f}, EM={avg_em:.4f}, F1={avg_f1:.4f}, Bleu1={avg_bleu:.4f}, count={count}"
        )

    final_overall_report = {
        "Macro_Average_Recall": sum(total_all_recalls) / len(total_all_recalls) if total_all_recalls else 0.0,
        "ExactMatch": sum(total_example_em) / len(total_example_em) if total_example_em else 0.0,
        "F1": sum(total_example_f1) / len(total_example_f1) if total_example_f1 else 0.0,
        "Bleu1": sum(total_example_bleu) / len(total_example_bleu) if total_example_bleu else 0.0,
        "total_questions": len(total_categories)
    }
    
    logger.info("--- Aggregated Overall Metrics (All Samples) ---")
    logger.info(f"Overall results: {json.dumps(final_overall_report, indent=2)}")


    metrics_path = os.path.join(base_save_dir, "aggregated_eval_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "overall": final_overall_report,
                "per_category": final_per_category_report,
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved aggregated evaluation metrics to {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save aggregated metrics. Error: {e}")


if __name__ == "__main__":
    main()
