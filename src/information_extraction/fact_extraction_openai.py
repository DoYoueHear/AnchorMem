import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from ..llm.openai_gpt import CacheOpenAI
from ..prompts import PromptTemplateManager
from ..utils.llm_utils import fix_broken_generated_json
from ..utils.logging_utils import get_logger
from ..utils.misc_utils import FactRawOutput

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def extract_fact_list(llm_output: str) -> List[str]:
    if not llm_output:
        return []

    text = llm_output.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)

    match = re.search(r"\[(.*)\]", text, re.DOTALL)
    list_content = match.group(1) if match else text

    try:
        candidate = f"[{list_content}]" if match else text
        data = json.loads(candidate)
        if _is_valid_str_list(data):
            return data
    except json.JSONDecodeError:
        pass

    try:
        candidate = f"[{list_content}]" if match else text
        data = ast.literal_eval(candidate)
        if _is_valid_str_list(data):
            return data
    except (ValueError, SyntaxError):
        pass

    return _fallback_regex_parsing(list_content)


def extract_event_list(llm_output: str) -> List[str]:
    if not llm_output:
        return []

    text = llm_output.strip().replace("<|COMPLETE|>", "").strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)

    events = []
    for raw_event in re.findall(r"\((.*?)\)", text, re.DOTALL):
        event = raw_event.strip()
        if event:
            events.append(event)
    return events


def _is_valid_str_list(data) -> bool:
    return isinstance(data, list) and all(isinstance(i, str) for i in data)


def _fallback_regex_parsing(text_content: str) -> List[str]:
    results = []

    quoted_items = re.findall(r'(["\'])(.*?)\1', text_content, re.DOTALL)
    if quoted_items:
        for _, content in quoted_items:
            content = content.strip()
            if content:
                results.append(content)
        return results

    raw_items = text_content.split(',')
    for item in raw_items:
        clean_item = item.strip().strip('"\'')
        if clean_item:
            results.append(clean_item)

    return results


class FactExtractor:
    def __init__(self, llm_model: CacheOpenAI):
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user"}
        )
        self.llm_model = llm_model

    def extract_facts(self, chunk_key: str, passage: str) -> FactRawOutput:
        fact_input_message = self.prompt_template_manager.render(name="fact", passage=passage)
        raw_response = ""
        metadata = {}
        try:
            raw_response, metadata, cache_hit = self.llm_model.infer(messages=fact_input_message)
            metadata["cache_hit"] = cache_hit
            if metadata.get("finish_reason") == "length":
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            facts = extract_fact_list(real_response)
        except Exception as e:
            logger.warning(e)
            metadata.update({"error": str(e)})
            return FactRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                facts=[],
                metadata=metadata,
            )

        return FactRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            facts=facts,
            metadata=metadata,
        )

    def batch_extract_facts(self, chunks: Dict[str, ChunkInfo]) -> Dict[str, FactRawOutput]:
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        max_workers = 16
        fact_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            fact_futures = {
                executor.submit(self.extract_facts, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(fact_futures), total=len(fact_futures), desc="Fact Extraction")
            for future in pbar:
                result = future.result()
                fact_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get("prompt_tokens", 0)
                total_completion_tokens += metadata.get("completion_tokens", 0)
                if metadata.get("cache_hit"):
                    num_cache_hit += 1

                pbar.set_postfix(
                    {
                        "total_prompt_tokens": total_prompt_tokens,
                        "total_completion_tokens": total_completion_tokens,
                        "num_cache_hit": num_cache_hit,
                    }
                )

        return {res.chunk_id: res for res in fact_results_list}
