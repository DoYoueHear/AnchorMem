import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import List, Set, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import re
import time
from sentence_transformers import util
import nltk
from nltk.corpus import stopwords

from src.llm import _get_llm_class, BaseLLM
from src.embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from embedding_store import EmbeddingStore


from src.information_extraction import FactExtractor
from src.information_extraction.fact_extraction_openai import extract_event_list
from src.utils.llm_utils import fix_broken_generated_json
from src.evaluation.qa_eval import QAExactMatch, QAF1Score
from src.prompts.linking import get_query_instruction
from src.prompts.prompt_template_manager import PromptTemplateManager
from src.utils.misc_utils import (
    QuerySolution,
    FactRawOutput,
    compute_mdhash_id,
    flatten_facts_str,
    min_max_normalize,
    reformat_fact_results,
)
from src.utils.logging_utils import setup_logger
from src.utils.config_utils import BaseConfig

try:
    STANDARD_STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STANDARD_STOP_WORDS = set(stopwords.words('english'))

# logger = logging.getLogger(__name__)

class AnchorMem:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific AnchorMem instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            fact_extractor (FactExtractor): The fact extraction module.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            fact_results_path (str): The file path for storing extracted fact and event results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        self.logger = setup_logger(os.path.join(self.global_config.save_dir, "AnchorMem.log"))

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        self.logger.debug(f"AnchorMem init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            self.logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)
        self.fact_extractor = FactExtractor(llm_model=self.llm_model)

        self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
            embedding_model_name=self.global_config.embedding_model_name
        )(
            global_config=self.global_config,
            embedding_model_name=self.global_config.embedding_model_name,
        )
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')
        self.event_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "event_embeddings"),
                                                   self.global_config.embedding_batch_size, 'event')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        llm_name_for_path = self.global_config.llm_name.replace("/", "_")
        self.fact_results_path = os.path.join(
            self.global_config.save_dir,
            f"fact_results_{llm_name_for_path}.json",
        )
        self.legacy_fact_cache_path = os.path.join(
            self.global_config.save_dir,
            f"openie_results_ner_{llm_name_for_path}.json",
        )
        self.legacy_event_cache_path = os.path.join(
            self.global_config.save_dir,
            f"openie_results_event_{llm_name_for_path}.json",
        )


        self.ready_to_retrieve = False
        
        self.cost_stats = {
            'summary': {'prompt': 0, 'completion': 0, 'calls': 0, 'time': 0.0}, # 对应 Indexing/Fact Extraction
            'update': {'prompt': 0, 'completion': 0, 'calls': 0, 'time': 0.0},  # 对应 Indexing/Event
            'qa': {'prompt': 0, 'completion': 0, 'calls': 0, 'time': 0.0},      # 对应 Generation
            'total': {'prompt': 0, 'completion': 0, 'calls': 0, 'time': 0.0}
        }
        self.start_time = time.time()

    def index(self, docs: List[str]):
        """
        Indexes the given documents for AnchorMem by extracting facts from chunks,
        grouping related facts, and generating event memories for retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """
        index_start_time = time.time()

        self.logger.info(f"Indexing Documents")
        self.logger.info("Extracting facts")

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        fact_results_payload, chunk_keys_to_process = self.load_existing_fact_results(chunk_to_rows.keys())
        all_fact_info = fact_results_payload["docs"]
        cached_event_results = fact_results_payload["events"]
        new_fact_rows = {k: chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_fact_results_dict = self.fact_extractor.batch_extract_facts(new_fact_rows)
            for res_dict in [new_fact_results_dict]:
                for item in res_dict.values():
                    if hasattr(item, 'metadata'):
                        self.cost_stats['summary']['prompt'] += item.metadata.get('prompt_tokens', 0)
                        self.cost_stats['summary']['completion'] += item.metadata.get('completion_tokens', 0)
                        self.cost_stats['summary']['calls'] += 1
            
            self.merge_fact_results(all_fact_info, new_fact_rows, new_fact_results_dict)

        fact_results_dict = reformat_fact_results(all_fact_info)

        assert len(chunk_to_rows) == len(fact_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(fact_results_dict): {len(fact_results_dict)}"

        chunk_ids = list(chunk_to_rows.keys())
        chunk_facts = [[fact for fact in fact_results_dict[chunk_id].facts] for chunk_id in chunk_ids]
        facts = flatten_facts_str(chunk_facts)

        self.logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings(facts)

        self.logger.info("Constructing fact-to-chunk mappings")

        self.fact_to_chunk_ids = defaultdict(set)
    
        all_facts_str = []
        all_chunk_ids = []
        
        print("Processing chunks and preparing data...")
        for chunk_id, facts_in_chunk in zip(chunk_ids, chunk_facts):
            for fact_str in facts_in_chunk:
                fact_id = self.fact_embedding_store.text_to_hash_id[fact_str] 
                self.fact_to_chunk_ids[fact_id].add(chunk_id)

                all_facts_str.append(fact_str)
                all_chunk_ids.append(chunk_id)

        llm_contexts = self._generate_fact_groups(all_facts_str, all_chunk_ids)

        if cached_event_results:
            self.load_cached_events(cached_event_results)
            print(f"Successfully loaded {len(cached_event_results)} events.")
            event_results = cached_event_results
        else:
            event_results = self.generate_and_map_events(llm_contexts)

        self.attach_events_to_fact_results(all_fact_info, event_results)

        if self.global_config.save_fact_results:
            self.save_fact_results(all_fact_info, event_results)

        self.cost_stats['update']['time'] += time.time() - index_start_time


    def generate_and_map_events(self, llm_contexts: List[Dict[str, Any]]):
        tasks = []
        self.fact_to_event_ids = defaultdict(set)

        for idx, context in enumerate(llm_contexts):
            main_text = context["main_fact"]["text"]
            supporting_blocks = []
            
            related_fact_texts = [main_text] 
            
            for sup in context["supporting_facts"]:
                s_text = sup["text"]
                related_fact_texts.append(s_text)
                
                chunk_id = sup["chunk_id"]
                chunk_content = self.chunk_embedding_store.get_row(chunk_id)["content"]
                supporting_blocks.append(f"[Topic]: {s_text}\n[Dialogue]: {chunk_content}")
            
            all_related_fact_texts = "\n-".join(related_fact_texts)
            full_prompt = f"Focus Topics:\n-{all_related_fact_texts}\n" + "Source Contexts:\n" + "\n---\n".join(supporting_blocks)
            
            related_fact_ids = [
                compute_mdhash_id(content=ft, prefix="fact-") 
                for ft in related_fact_texts
            ]
            
            tasks.append({
                "id": idx,
                "prompt": full_prompt,
                "fact_ids": related_fact_ids
            })

        return self.batch_process_events(tasks)



    def _generate_fact_groups(self, all_facts_str, all_chunk_ids):
        """
        封装后的 'method demo' 核心逻辑
        """

        self.sim_threshold = self.global_config.fact_sim_threshold
        self.related_fact_top_k = self.global_config.related_fact_top_k
        self.min_diff_threshold = 3

        total_facts = len(all_facts_str)
        
        
        embeddings = self.embedding_model.batch_encode(all_facts_str, convert_to_tensor=True, show_progress_bar=True)

        # 2. Computing Similarity
        print("Computing Similarity Matrix...")
        cos_sim_matrix = util.cos_sim(embeddings, embeddings)
        cos_sim_matrix.fill_diagonal_(0) # 排除自身的完全匹配
        sim_matrix_np = cos_sim_matrix.cpu().numpy()

        raw_fact_groups = []

        # 3. Phase 1: Building Fact Groups
        print(f"Phase 1: Building Fact Groups (Threshold={self.sim_threshold})...")
        for i in tqdm(range(total_facts), mininterval=1.0):
            candidate_indices = np.where(sim_matrix_np[i] > self.sim_threshold)[0]
            
            if len(candidate_indices) == 0:
                continue
                
            candidate_scores = sim_matrix_np[i][candidate_indices]
            sorted_order = np.argsort(candidate_scores)[::-1]
            sorted_indices = candidate_indices[sorted_order]
            sorted_scores = candidate_scores[sorted_order]
            
            valid_matches = []
            seen_chunks = {all_chunk_ids[i]}
            
            group_score_sum = 0
            
            for rank, match_idx in enumerate(sorted_indices):
                match_chunk = all_chunk_ids[match_idx]
                
                if match_chunk in seen_chunks:
                    continue
                    
                valid_matches.append(match_idx)
                seen_chunks.add(match_chunk)
                group_score_sum += sorted_scores[rank]
                
                if len(valid_matches) >= self.related_fact_top_k:
                    break
            
            if len(valid_matches) > 0:
                raw_fact_groups.append({
                    "query_idx": i,
                    "match_indices": valid_matches,
                    "score_sum": group_score_sum,
                    "total_items": 1 + len(valid_matches) 
                })

        raw_fact_groups.sort(key=lambda x: (x['total_items'], x['score_sum']), reverse=True)
        
        final_groups = []

        print("Phase 2: Inter-Group Deduplication...")
        for group in tqdm(raw_fact_groups):
            current_ids = set([group['query_idx']] + group['match_indices'])
            
            is_redundant = False
            
            for kept_group in final_groups:
                kept_ids = set([kept_group['query_idx']] + kept_group['match_indices'])
                
                diff_ids = current_ids - kept_ids
                
                if len(diff_ids) < self.min_diff_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                final_groups.append(group)

        print(f"Reduced from {len(raw_fact_groups)} to {len(final_groups)} unique contexts.")

        llm_contexts = []
        for group in final_groups:
            query_idx = group['query_idx']
            matches = group['match_indices']
            
            context_pack = {
                "main_fact": {
                    "text": all_facts_str[query_idx],
                    "chunk_id": all_chunk_ids[query_idx],
            
                },
                "supporting_facts": []
            }
            
            for m_idx in matches:
                context_pack["supporting_facts"].append({
                    "text": all_facts_str[m_idx],
                    "chunk_id": all_chunk_ids[m_idx],
                    "score": float(sim_matrix_np[query_idx][m_idx])
                })
            
            llm_contexts.append(context_pack)
            
        return llm_contexts

        
    def retrieve(self, queries: List[str]) -> List[QuerySolution]:
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects() 
        
        retrieval_results = []

        for query in tqdm(queries, desc="Retrieving", total=len(queries)):
            query_fact_scores, query_event_scores = self.get_fact_scores(query)
            _, top_k_facts, _, candidate_events = self.rerank_facts(query_fact_scores, query_event_scores)

            top_k_fact_ids = [(compute_mdhash_id(content=top_k_fact, prefix="fact-")) for top_k_fact in top_k_facts]
            initial_chunk_ids = set()

            for fact_id in top_k_fact_ids:
                initial_chunk_ids.update(self.fact_to_chunk_ids.get(fact_id, set()))
                
            context_chunks = []
            for chunk_id in list(initial_chunk_ids):
                chunk_content = self.chunk_embedding_store.get_row(chunk_id)['content']
                context_chunks.append(chunk_content)
            for event in candidate_events:
                context_chunks.append(event)

            retrieval_results.append(QuerySolution(question=query, docs=context_chunks, topk_facts=top_k_facts))

        return retrieval_results
    

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the AnchorMem framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)
        # return "_", "_", "_"
        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Diagnostics to avoid silent termination and mismatches
        try:
            self.logger.info(f"QA completed: solutions={len(queries_solutions)}, responses={len(all_response_message)}, metadata={len(all_metadata)}")
        except Exception as e:
            self.logger.error(f"Logging QA sizes failed: {str(e)}")

        # Evaluating QA
        if gold_answers is not None:
            # Align lengths defensively
            preds = [qa_result.answer for qa_result in queries_solutions]
            if len(preds) != len(gold_answers):
                self.logger.warning(f"Predictions ({len(preds)}) and gold_answers ({len(gold_answers)}) length mismatch; aligning to min length.")
                min_len = min(len(preds), len(gold_answers))
                preds = preds[:min_len]
                gold_answers = gold_answers[:min_len]
            try:
                overall_qa_em_result, _ = qa_em_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers, predicted_answers=preds, aggregation_fn=np.max)
                overall_qa_f1_result, _ = qa_f1_evaluator.calculate_metric_scores(
                    gold_answers=gold_answers, predicted_answers=preds, aggregation_fn=np.max)
            except Exception as e:
                self.logger.error(f"QA evaluation failed: {str(e)}")
                overall_qa_em_result, overall_qa_f1_result = {}, {}

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            self.logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata


    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        qa_start_time = time.time()

        #Running inference for QA
        all_qa_messages = []

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):

            # obtain the retrieved docs
            # retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]
            retrieved_passages = query_solution.docs
            # print("="*50)
            # # print(type(retrieved_passages))
            # # print(len(retrieved_passages))
            # # print(retrieved_passages[0])
            # print()
            # print("="*50)

            prompt_user = '# Memories: \n'
            for passage in retrieved_passages:
                prompt_user += f'{passage}\n'
            prompt_user += '# QUESTION: ' + query_solution.question + '\n# Answer: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                self.logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            all_qa_messages.append(
                self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user))      


        # Run QA inference concurrently with robust error handling
        max_workers = 10
        all_qa_results = [None] * len(all_qa_messages)
        errors: List[int] = []
        self.logger.info(f"Starting QA inference for {len(all_qa_messages)} prompts with max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.llm_model.infer, msg): idx for idx, msg in enumerate(all_qa_messages)}
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="QA Reading"):
                idx = future_to_idx[future]
                try:
                    all_qa_results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"QA inference failed at index {idx}: {str(e)}")
                    errors.append(idx)
                    # Fallback to empty response to keep downstream logic intact
                    all_qa_results[idx] = ("", {}, False)

        if errors:
            self.logger.warning(f"QA inference encountered {len(errors)} failures; indices: {errors[:10]}{'...' if len(errors) > 10 else ''}")
        self.logger.info(f"Completed QA inference. Results={len(all_qa_results)}, Errors={len(errors)}")

        try:
            all_response_message, all_metadata, _ = zip(*all_qa_results)
            all_response_message, all_metadata = list(all_response_message), list(all_metadata)
            for meta in all_metadata:
                self.cost_stats['qa']['prompt'] += meta.get('prompt_tokens', 0)
                self.cost_stats['qa']['completion'] += meta.get('completion_tokens', 0)
                self.cost_stats['qa']['calls'] += 1
            self.cost_stats['qa']['time'] += time.time() - qa_start_time

        except Exception as e:
            self.logger.error(f"Error unpacking QA results: {str(e)}")
            # Graceful fallback: coerce results into expected lists
            all_response_message = []
            all_metadata = []
            for r in all_qa_results:
                if isinstance(r, tuple):
                    all_response_message.append(r[0] if len(r) > 0 else "")
                    all_metadata.append(r[1] if len(r) > 1 else {})
                else:
                    all_response_message.append(str(r))
                    all_metadata.append({})

        #Process responses and extract predicted answers.
        queries_solutions = []
        safe_len = min(len(queries), len(all_response_message))
        self.logger.info(f"Extracting answers for {safe_len}/{len(queries)} queries")
        for query_solution_idx in tqdm(range(safe_len), desc="Extraction Answers from LLM Response"):
            query_solution = queries[query_solution_idx]
            response_content = all_response_message[query_solution_idx]

            pattern = re.compile(r'</think>\s*([\s\S]*)')
            m = pattern.search(response_content)
            if m:
                response_content = m.group(1)

            query_solution.answer = response_content.strip()
            queries_solutions.append(query_solution)

        if safe_len < len(queries):
            self.logger.warning(f"Missing QA responses for {len(queries) - safe_len} queries; filling empty answers.")
            for i in range(safe_len, len(queries)):
                q = queries[i]
                q.answer = ""
                queries_solutions.append(q)

        self.logger.info(f"QA extraction completed. Prepared {len(queries_solutions)} answers.")

        return queries_solutions, all_response_message, all_metadata


    def _llm_event_worker(self, task_id, prompt, related_fact_ids):
        try:
            input_message = self.prompt_template_manager.render(name='event', passage=prompt)

            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=input_message
            )
            
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            extracted_events = extract_event_list(real_response)
            
            return {
                "status": "success",
                "task_id": task_id,
                "events": extracted_events,
                "related_fact_ids": related_fact_ids,
                "metadata": metadata,
                "cache_hit": cache_hit
            }

        except Exception as e:
            return {
                "status": "error",
                "task_id": task_id,
                "error": str(e),
                "related_fact_ids": related_fact_ids
            }

    def batch_process_events(self, tasks):
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0
        
        all_event_results = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for task in tasks:
                future = executor.submit(
                    self._llm_event_worker, 
                    task['id'], 
                    task['prompt'], 
                    task['fact_ids']
                )
                futures.append(future)

            pbar = tqdm(as_completed(futures), total=len(futures), desc="Event Extraction")
            
            for future in pbar:
                result = future.result()
                
                if result["status"] == "error":
                    continue
                
                events = result["events"]
                fact_ids = result["related_fact_ids"]
                
                for event_content in events:
                    event_text = event_content if isinstance(event_content, str) else str(event_content)
                    event_id = compute_mdhash_id(content=event_text, prefix="event-")

                    for f_id in fact_ids:
                        self.fact_to_event_ids[f_id].add(event_id)

                    all_event_results.append({
                        "idx": event_id,
                        "event": event_text,
                        "source_facts": fact_ids
                    })

                meta = result.get("metadata", {})
                total_prompt_tokens += meta.get('prompt_tokens', 0)
                total_completion_tokens += meta.get('completion_tokens', 0)
                if result.get("cache_hit"):
                    num_cache_hit += 1
                
                pbar.set_postfix({
                    'tk_in': total_prompt_tokens,
                    'tk_out': total_completion_tokens,
                    'hits': num_cache_hit
                })

        self.cost_stats['update']['prompt'] = total_prompt_tokens
        self.cost_stats['update']['completion'] = total_completion_tokens
        self.cost_stats['update']['calls'] = len(tasks)

        self.event_embedding_store.insert_strings([item['event'] for item in all_event_results])
        return all_event_results

    def load_cached_events(self, cached_event_results: List[dict]) -> None:
        self.fact_to_event_ids = defaultdict(set)
        for item in cached_event_results:
            event_id = item["idx"]
            for fact_id in item.get("source_facts", []):
                self.fact_to_event_ids[fact_id].add(event_id)
        self.event_embedding_store.insert_strings([item["event"] for item in cached_event_results])

    def load_existing_fact_results(self, chunk_keys: List[str]) -> Tuple[Dict[str, List[dict]], Set[str]]:
        chunk_keys_to_save = set()
        fact_results_payload = {"docs": [], "events": []}

        if self.global_config.force_fact_extraction_from_scratch:
            return fact_results_payload, set(chunk_keys)

        source_path = None
        if os.path.isfile(self.fact_results_path):
            source_path = self.fact_results_path
        elif os.path.isfile(self.legacy_fact_cache_path):
            source_path = self.legacy_fact_cache_path

        if source_path is None:
            return fact_results_payload, set(chunk_keys)

        with open(source_path, "r", encoding="utf-8") as f:
            loaded_results = json.load(f)

        fact_docs = loaded_results.get("docs", [])
        normalized_fact_docs = []
        for fact_info in fact_docs:
            normalized_fact_docs.append({
                "idx": compute_mdhash_id(fact_info["passage"], "chunk-"),
                "passage": fact_info["passage"],
                "extracted_facts": fact_info.get("extracted_facts", fact_info.get("extracted_entities", [])),
                "extracted_events": fact_info.get("extracted_events", fact_info.get("extracted_triples", [])),
            })

        event_results = loaded_results.get("events", [])
        if not event_results and os.path.isfile(self.legacy_event_cache_path):
            with open(self.legacy_event_cache_path, "r", encoding="utf-8") as f:
                legacy_event_results = json.load(f)
            event_results = legacy_event_results.get("docs", [])

        fact_results_payload = {"docs": normalized_fact_docs, "events": event_results}
        existing_fact_keys = {info["idx"] for info in normalized_fact_docs}
        for chunk_key in chunk_keys:
            if chunk_key not in existing_fact_keys:
                chunk_keys_to_save.add(chunk_key)

        return fact_results_payload, chunk_keys_to_save

    def merge_fact_results(
        self,
        all_fact_info: List[dict],
        chunks_to_save: Dict[str, dict],
        fact_results_dict: Dict[str, FactRawOutput],
    ) -> List[dict]:
        for chunk_key, row in chunks_to_save.items():
            passage = row["content"]
            try:
                chunk_fact_info = {
                    "idx": chunk_key,
                    "passage": passage,
                    "extracted_facts": fact_results_dict[chunk_key].facts,
                    "extracted_events": [],
                }
            except Exception as e:
                self.logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_fact_info = {
                    "idx": chunk_key,
                    "passage": passage,
                    "extracted_facts": [],
                    "extracted_events": [],
                }
            all_fact_info.append(chunk_fact_info)

        return all_fact_info

    def attach_events_to_fact_results(self, all_fact_info: List[dict], event_results: List[dict]) -> None:
        fact_info_by_chunk_id = {item["idx"]: item for item in all_fact_info}

        for item in all_fact_info:
            item["extracted_events"] = item.get("extracted_events", [])

        for event_item in event_results:
            event_text = event_item["event"]
            for fact_id in event_item.get("source_facts", []):
                for chunk_id in self.fact_to_chunk_ids.get(fact_id, set()):
                    chunk_result = fact_info_by_chunk_id.get(chunk_id)
                    if chunk_result is None:
                        continue
                    if event_text not in chunk_result["extracted_events"]:
                        chunk_result["extracted_events"].append(event_text)

    def save_fact_results(self, all_fact_info: List[dict], event_results: List[dict]) -> None:
        fact_result_dict = {
            "docs": all_fact_info,
            "events": event_results,
        }
        with open(self.fact_results_path, "w", encoding="utf-8") as f:
            json.dump(fact_result_dict, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Fact results saved to {self.fact_results_path}")



    def prepare_retrieval_objects(self):
        """
        Prepares in-memory retrieval objects for chunks, facts, and events.
        """

        self.logger.info("Preparing for fast retrieval.")

        self.logger.info("Loading keys.")
        self.passage_node_keys: List = list(self.chunk_embedding_store.get_all_ids())
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())
        self.event_node_keys: List = list(self.event_embedding_store.get_all_ids())

        self.logger.info("Loading embeddings.")
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.passage_node_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))
        self.event_embeddings = np.array(self.event_embedding_store.get_embeddings(self.event_node_keys))
        
        self.ready_to_retrieve = True
        
    def get_fact_scores(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        
        query_embedding = self.embedding_model.batch_encode([query],
                                                            instruction=get_query_instruction('query_to_atomic'),
                                                            norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            self.logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([]), np.array([])
            
        try:
            # print("%"*50)
            # print(f"self.fact_embeddings: {self.fact_embeddings.shape}")
            # print(f"query_embedding: {query_embedding.shape}")
            # print("%"*50)
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            
            if len(self.event_embeddings) == 0:
                query_event_scores = np.array([])
            else:
                query_event_scores = np.dot(self.event_embeddings, query_embedding.T)
                query_event_scores = np.squeeze(query_event_scores) if query_event_scores.ndim == 2 else query_event_scores
                query_event_scores = min_max_normalize(query_event_scores)
            return query_fact_scores, query_event_scores
        except Exception as e:
            self.logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([]), np.array([])
    
    def rerank_facts(self, query_fact_scores: np.ndarray, query_event_scores: np.ndarray) -> Tuple[List[int], List[str], dict, List[str]]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of linked facts.
                - top_k_facts:
        """
        link_top_k: int = self.global_config.linking_top_k

        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            self.logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}, []
            
        if len(query_fact_scores) <= link_top_k:
            candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
        else:
            candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()

        if len(query_event_scores) == 0 or len(self.event_node_keys) == 0:
            candidate_event_indices = []
        elif len(query_event_scores) <= link_top_k:
            candidate_event_indices = np.argsort(query_event_scores)[::-1].tolist()
        else:
            candidate_event_indices = np.argsort(query_event_scores)[-link_top_k:][::-1].tolist()

        real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
        fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
        candidate_facts = [fact_row_dict[id]['content'] for id in real_candidate_fact_ids]
        
        real_candidate_event_ids = [self.event_node_keys[idx] for idx in candidate_event_indices]
        event_row_dict = self.event_embedding_store.get_rows(real_candidate_event_ids)
        candidate_events = [event_row_dict[id]['content'] for id in real_candidate_event_ids]
        
        
        rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': candidate_facts}
        
        return candidate_fact_indices, candidate_facts, rerank_log, candidate_events
            
    def save_cost_summary(self, filename="cost_summary.json"):
        """
        保存成本统计数据到 save_dir 目录下的 JSON 文件中
        """
        total_prompt = sum(self.cost_stats[k]['prompt'] for k in ['summary', 'update', 'qa'])
        total_completion = sum(self.cost_stats[k]['completion'] for k in ['summary', 'update', 'qa'])
        total_calls = sum(self.cost_stats[k]['calls'] for k in ['summary', 'update', 'qa'])
        total_time = time.time() - self.start_time

        output_data = {
            "stages": self.cost_stats,
            "total": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "calls": total_calls,
                "time_seconds": total_time
            },
            "meta": {
                "llm_name": self.global_config.llm_name,
                "dataset": self.global_config.dataset,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        save_path = os.path.join(self.global_config.save_dir, filename)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            self.logger.info(f"Cost summary saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save cost summary: {e}")
