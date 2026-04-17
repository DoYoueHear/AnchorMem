import os
import json
import argparse
import logging
import dataclasses
from typing import List, Optional, Dict, Any
from collections import defaultdict
import numpy as np
import re
import asyncio
from openai import AsyncOpenAI


try:
    client = AsyncOpenAI()
except Exception:
    print("Warning: OpenAI client could not be initialized. Please set API key or adjust client initialization.")


def extract_json(text):
    """
    Extracts JSON content from a string, removing enclosing triple backticks and optional 'json' tag if present.
    If no code block is found, returns the text as-is.
    """
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
    return json_str


ACCURACY_PROMPT = """
Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a ’gold’ (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


async def evaluate_llm_judge(
    question: str, 
    gold_answer: Any, 
    generated_answer: str, 
    client_obj: AsyncOpenAI, 
    model_name: str,
    semaphore: asyncio.Semaphore
) -> int:
    """Evaluate the generated answer against the gold answer using an LLM judge asynchronously."""
    if client_obj is None:
        raise ValueError("OpenAI client not configured.")
        
    if isinstance(gold_answer, list):
        gold_answer_str = gold_answer[0]
    else:
        gold_answer_str = str(gold_answer)

    prompt = ACCURACY_PROMPT.format(
        question=question, gold_answer=gold_answer_str, generated_answer=generated_answer
    )

    # 使用 semaphore 限制并发数
    async with semaphore:
        try:
            response = await client_obj.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            label = json.loads(extract_json(content))["label"].strip().upper()
            return 1 if label == "CORRECT" else 0
        except Exception as e:
            logging.getLogger('locomo_eval_judge').error(f"LLM Judge API Error for Q: {question[:30]}... Error: {e}")
            return 0


async def process_single_qa(
    dataset_qa: Dict, 
    cached_solution: Dict, 
    client_obj: AsyncOpenAI, 
    model_name: str, 
    semaphore: asyncio.Semaphore
) -> Dict:
    """Wrapper to process a single QA pair and return the result entry."""
    question = dataset_qa["question"]
    gold_answers = dataset_qa["gold_answers"]
    category = dataset_qa["category"]
    generated_answer = cached_solution.get("answer")
    
    llm_label_score = await evaluate_llm_judge(
        question=question,
        gold_answer=gold_answers[0] if gold_answers else "",
        generated_answer=generated_answer,
        client_obj=client_obj,
        model_name=model_name,
        semaphore=semaphore
    )
    
    return {
        "question": question,
        "gt_answer": gold_answers,
        "response": generated_answer,
        "category": category,
        "llm_label": "CORRECT" if llm_label_score == 1 else "WRONG",
        "llm_label_score": llm_label_score,
    }



@dataclasses.dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    answer: str = None # LLM 生成的答案
    query_triple: str = None
    gold_answers: List[str] = None # 注意：缓存中可能没有这个字段，需要从数据集补充
    gold_docs: Optional[List[str]] = None
    topk_facts: Optional[List[str]] = None

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('locomo_eval_judge')
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


async def main():
    parser = argparse.ArgumentParser(description="Evaluate cached RAG results using LLM judge")
    parser.add_argument('--dataset', type=str, default='locomo', help='Dataset name (used for prompts/config)')
    parser.add_argument('--dataset_path', type=str, default='your/path/to/locomo10.json',
                        help='Path to LoCoMo10 dataset JSON for evaluation')
    parser.add_argument('--base_save_dir', type=str, default='outputs', help='Base directory where sample_* directories and caches are saved')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Exact experiment directory produced by main.py. If provided, this overrides base_save_dir path construction.')
    parser.add_argument('--llm_judge_model', type=str, default='Qwen2.5-32B', help='Model to use for the LLM Judge')
    parser.add_argument('--llm_base_url', type=str, default='http://localhost:8080/v1', help='LLM base URL for the LLM Judge (e.g., for custom endpoints)')
    parser.add_argument('--fact_sim_threshold', type=float, default=0.85, help='Fact similarity threshold for retrieval')
    parser.add_argument('--related_fact_top_k', type=int, default=5, help='Number of related facts to retrieve')
    parser.add_argument('--exp_id', type=str, default='v1', help='')
    parser.add_argument('--retri_topk', type=int, default=10, help='Top-K for retrieval')
    parser.add_argument('--concurrency', type=int, default=20, help='Max concurrent API requests')
    
    args = parser.parse_args()
    dataset_name = args.dataset
    
    if args.results_dir:
        base_save_dir = args.results_dir
    else:
        base_save_dir = args.base_save_dir
        if base_save_dir == 'outputs':
            base_save_dir = base_save_dir + '/' + dataset_name + '-' + args.llm_judge_model
        else:
            base_save_dir = base_save_dir + '_' + dataset_name  + '-' + args.llm_judge_model

    if not os.path.exists(base_save_dir):
        print(f"Error: Base save directory not found at {base_save_dir}")
        return

    log_file = os.path.join(base_save_dir, "llm_judge_aggregated.log")
    logger = setup_logger(log_file=log_file)
    logger.info(f"Starting LLM Judge evaluation for results in: {base_save_dir}")

    # 1. 初始化 LLM Judge 客户端
    judge_client = None
    if args.llm_base_url:
        try:
            judge_client = AsyncOpenAI(base_url=args.llm_base_url)
            logger.info(f"Initialized Judge Client with base_url: {args.llm_base_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Judge Client with custom base_url. Error: {e}")
            return
    elif client:

        judge_client = client
        logger.info("Using globally imported OpenAI client for judging.")
    else:
        logger.error("No valid OpenAI client could be configured for LLM Judge.")
        return

    # 2. 加载数据集获取 Gold Answers
    try:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            locomo10_samples = json.load(f)
        logger.info(f"Loaded LoCoMo10 dataset from {args.dataset_path} for gold answers.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 3. 聚合指标存储
    final_metrics_per_category = defaultdict(lambda: {"llm_judge_scores": []})
    total_categories = []
    total_llm_judge_scores = []

    # 4. 遍历每个样本目录
    for sample_index, sample_data in enumerate(locomo10_samples):
        sample_save_dir = os.path.join(base_save_dir, f"sample_{sample_index}")
        cache_file_path = os.path.join(sample_save_dir, "queries_solutions.json")

        if not os.path.exists(cache_file_path):
            logger.warning(f"[Sample {sample_index}] Cache file not found at {cache_file_path}. Skipping.")
            continue
        
        logger.info(f"\n{'='*30} Processing Sample {sample_index+1} / {len(locomo10_samples)} for LLM Judge {'='*30}")

        semaphore = asyncio.Semaphore(args.concurrency)
        qa_pairs_from_dataset = []
        for qa in sample_data.get("qa", []):
            cat = int(qa.get("category"))
            if cat == 5: # 跳过 category 5
                continue
            gold_ans = qa.get("answer")
            gold_answers_list = [str(a) for a in (gold_ans if isinstance(gold_ans, list) else [gold_ans])]
            qa_pairs_from_dataset.append({
                "question": qa["question"],
                "gold_answers": gold_answers_list,
                "category": cat
            })
            
        if not qa_pairs_from_dataset:
            logger.warning(f"[Sample {sample_index}] No QA pairs (excluding Cat 5) found in dataset. Skipping.")
            continue


        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                loaded_solutions = json.load(f)
            
            # 确保数据集和缓存中的 QA 对数量匹配
            if len(loaded_solutions) != len(qa_pairs_from_dataset):
                logger.error(f"[Sample {sample_index}] Mismatch: Dataset QA count ({len(qa_pairs_from_dataset)}) != Cached Solutions count ({len(loaded_solutions)}). Skipping sample.")
                continue

            tasks = []
            for i in range(len(loaded_solutions)):
                dataset_qa = qa_pairs_from_dataset[i]
                cached_solution = loaded_solutions[i]
                
                # 创建任务
                task = process_single_qa(
                    dataset_qa=dataset_qa,
                    cached_solution=cached_solution,
                    client_obj=judge_client,
                    model_name=args.llm_judge_model,
                    semaphore=semaphore
                )
                tasks.append(task)
            
            # 等待当前 Sample 所有任务完成
            sample_llm_judge_results = await asyncio.gather(*tasks)
            
            # 处理结果并聚合
            for res in sample_llm_judge_results:
                category = res["category"]
                score = res["llm_label_score"]
                question = res["question"]
                
                total_categories.append(category)
                total_llm_judge_scores.append(score)
                final_metrics_per_category[category]["llm_judge_scores"].append(score)
                
                # 简化日志输出，避免刷屏
                logger.info(f"[Sample {sample_index}] Cat {category} Q: {question[:30]}... -> {res['llm_label']}")

            # 保存结果
            sample_judge_output_path = os.path.join(sample_save_dir, "llm_judge_results.json")
            with open(sample_judge_output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_llm_judge_results, f, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            logger.info(f"[Sample {sample_index}] Saved results to {sample_judge_output_path}")

        except Exception as e:
            logger.error(f"[Sample {sample_index}] Error processing: {e}")
            continue


    logger.info(f"\n{'='*80}\nCalculating aggregated metrics...\n{'='*80}")


    if not total_categories:
        logger.error("No data collected across all samples for LLM Judge. Exiting.")
        return

    final_per_category_report = {}
    main_v1_report = {}
    
    main_v1_metrics_path = os.path.join(base_save_dir, "aggregated_eval_metrics.json")
    if os.path.exists(main_v1_metrics_path):
        try:
            with open(main_v1_metrics_path, "r", encoding="utf-8") as f:
                main_v1_report = json.load(f)
                final_per_category_report = main_v1_report.get("per_category", {})
                logger.info(f"Loaded existing metrics from {main_v1_metrics_path} for merging.")
        except Exception as e:
            logger.warning(f"Failed to load existing metrics from {main_v1_metrics_path}: {e}. Starting fresh report.")
            
    logger.info("--- Aggregated Per-Category Metrics (Including LLM Judge) ---")
    
    all_categories = sorted(list(set(total_categories)))

    for cat in all_categories:
        data = final_metrics_per_category[cat]
        count = len(data["llm_judge_scores"])
        if count == 0:
            continue
        
        avg_llm_judge_acc = sum(data["llm_judge_scores"]) / count

        # 更新或创建类别报告
        cat_report = final_per_category_report.get(str(cat), {})
        cat_report["LLM_Judge_Accuracy"] = avg_llm_judge_acc
        
        # 尝试保留旧的 count，否则使用新的
        if 'count' not in cat_report:
             cat_report["count"] = count
             
        final_per_category_report[str(cat)] = cat_report

        
        recall_str = f"Recall={cat_report.get('Recall', 0.0):.4f}" if 'Recall' in cat_report else ""
        em_str = f"EM={cat_report.get('ExactMatch', 0.0):.4f}" if 'ExactMatch' in cat_report else ""
        f1_str = f"F1={cat_report.get('F1', 0.0):.4f}" if 'F1' in cat_report else ""
        bleu_str = f"Bleu1={cat_report.get('Bleu1', 0.0):.4f}" if 'Bleu1' in cat_report else ""
        
        logger.info(
            f"Category {cat}: {recall_str}, {em_str}, {f1_str}, {bleu_str}, LLM_Judge_Accuracy={avg_llm_judge_acc:.4f}, count={cat_report.get('count', count)}"
        )


    final_overall_report = {}
    if 'overall' in main_v1_report:
        final_overall_report.update(main_v1_report['overall'])
        
    overall_llm_judge_acc = sum(total_llm_judge_scores) / len(total_llm_judge_scores) if total_llm_judge_scores else 0.0
    final_overall_report["LLM_Judge_Accuracy"] = overall_llm_judge_acc

    logger.info("--- Aggregated Overall Metrics (Including LLM Judge) ---")
    logger.info(f"Overall results: {json.dumps(final_overall_report, indent=2)}")

    metrics_path = os.path.join(base_save_dir, "aggregated_eval_metrics_with_judge.json")
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
    # main()
    asyncio.run(main())
