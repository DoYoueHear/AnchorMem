import json
from typing import List, Dict, Any


def load_locomo10(path: str) -> List[Dict[str, Any]]:
    """
    Load LoCoMo10 dataset JSON.

    Returns a list of samples, each with a key "qa" that is a list of QA dicts:
    {"question": str, "answer": str|int|list, "evidence": List[str], "category": int}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("locomo10.json root must be a list of samples")
    return data


def extract_doc_ids_from_evidence(samples: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Parse evidence entries like "D1:3" to collect doc ids and line indices.
    Returns mapping doc_id -> sorted unique line indices.
    """
    mapping: Dict[str, set] = {}
    for sample in samples:
        for qa in sample.get("qa", []):
            for ev in qa.get("evidence", []):
                if not isinstance(ev, str):
                    continue
                if ":" in ev and ev.startswith("D"):
                    doc_id, line_str = ev.split(":", 1)
                    try:
                        line_no = int(line_str)
                    except ValueError:
                        line_no = None
                    if doc_id not in mapping:
                        mapping[doc_id] = set()
                    if line_no is not None:
                        mapping[doc_id].add(line_no)
                else:
                    # fallback: treat whole string as a doc id
                    mapping.setdefault(ev, set())
    # convert sets to sorted lists
    return {doc_id: sorted(list(lines)) for doc_id, lines in mapping.items()}


def make_docs_from_samples(samples: List[Dict[str, Any]], per_qa: bool = True) -> List[str]:
    """
    Construct indexable text documents from LoCoMo10 samples.

    If per_qa=True, returns one short document per QA that includes the question,
    answer (as-is), evidence codes, and category. This uses only locomo10.json data.

    If per_qa=False, returns one document per sample by concatenating all QA entries
    of that sample together.
    """
    docs: List[str] = []
    if per_qa:
        for si, sample in enumerate(samples):
            for qi, qa in enumerate(sample.get("qa", [])):
                question = str(qa.get("question", "")).strip()
                answer = qa.get("answer")
                if isinstance(answer, list):
                    answer_str = ", ".join([str(a) for a in answer])
                else:
                    answer_str = str(answer)
                evidence = ", ".join([str(e) for e in qa.get("evidence", [])])
                category = qa.get("category", "")
                doc = (
                    f"Sample {si} QA {qi}\n"
                    f"Question: {question}\n"
                    f"Answer: {answer_str}\n"
                    f"Evidence: {evidence}\n"
                    f"Category: {category}"
                )
                docs.append(doc)
    else:
        for si, sample in enumerate(samples):
            lines: List[str] = [f"Sample {si}"]
            for qi, qa in enumerate(sample.get("qa", [])):
                question = str(qa.get("question", "")).strip()
                answer = qa.get("answer")
                if isinstance(answer, list):
                    answer_str = ", ".join([str(a) for a in answer])
                else:
                    answer_str = str(answer)
                evidence = ", ".join([str(e) for e in qa.get("evidence", [])])
                category = qa.get("category", "")
                lines.append(f"QA {qi} | Q: {question} | A: {answer_str} | Ev: {evidence} | Cat: {category}")
            docs.append("\n".join(lines))
    return docs


def load_locomo_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """
    Load LoCoMo corpus with conversation turns, expected fields include
    keys like: {"speaker": str, "text": str, "date_time": str, "idx": "D#:line"}
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    if not isinstance(corpus, list):
        raise ValueError("locomo_corpus.json must be a list of turn dicts")
    return corpus


# def make_docs_from_corpus_using_evidence(
#     samples: List[Dict[str, Any]],
#     corpus: List[Dict[str, Any]],
#     per_session: bool = True,
# ) -> List[str]:
#     """
#     Construct indexable docs from the full conversation corpus, but only include
#     sessions (e.g., "D1", "D2") that are referenced by evidence codes in locomo10.

#     - If per_session=True: one document per session id (e.g., D1), consisting of
#       all turns from that session concatenated as multi-line text.
#     - If per_session=False: one document per turn (i.e., per idx like "D1:3").
#     """
#     # Collect referenced doc_ids from evidence strings
#     referenced_doc_ids = set()
#     for sample in samples:
#         for qa in sample.get("qa", []):
#             for ev in qa.get("evidence", []):
#                 if isinstance(ev, str) and ":" in ev and ev.startswith("D"):
#                     doc_id = ev.split(":", 1)[0]  # e.g., "D1"
#                     referenced_doc_ids.add(doc_id)

#     # Build helper structures from corpus
#     # Map each turn's idx to its text, and group turns by session id
#     session_to_turns: Dict[str, List[str]] = {}
#     idx_to_text: Dict[str, str] = {}
#     for entry in corpus:
#         idx = str(entry.get("idx", ""))
#         speaker = str(entry.get("speaker", "")).strip()
#         text = str(entry.get("text", "")).strip()
#         if not idx or ":" not in idx:
#             continue
#         doc_id = idx.split(":", 1)[0]
#         turn_text = f"{speaker}: {text}"
#         idx_to_text[idx] = turn_text
#         session_to_turns.setdefault(doc_id, []).append(turn_text)

#     docs: List[str] = []
#     if per_session:
#         # Build one doc per referenced session id, concatenating all turns
#         for doc_id in sorted(referenced_doc_ids):
#             turns = session_to_turns.get(doc_id, [])
#             if not turns:
#                 # If no turns found, skip to avoid empty docs
#                 continue
#             doc_text = doc_id + "\n" + "\n".join(turns)
#             docs.append(doc_text)
#     else:
#         # Build one doc per referenced turn (idx) ONLY for turns referenced in evidence
#         referenced_indices = set()
#         for sample in samples:
#             for qa in sample.get("qa", []):
#                 for ev in qa.get("evidence", []):
#                     if isinstance(ev, str) and ":" in ev and ev.startswith("D"):
#                         referenced_indices.add(ev)
#         for idx in sorted(referenced_indices):
#             turn_text = idx_to_text.get(idx)
#             if turn_text:
#                 doc_id = idx.split(":", 1)[0]
#                 docs.append(doc_id + "\n" + turn_text)

#     return docs


# def make_docs_from_evidence_turns(
#     samples: List[Dict[str, Any]],
#     corpus: List[Dict[str, Any]],
# ) -> List[str]:
#     """
#     Construct one document per evidence turn (e.g., "D1:3") by matching locomo_corpus
#     entries whose `idx` equals the evidence code. Each doc is small and contains
#     just that single turn text, prefixed with its session id.
#     """
#     # Collect referenced indices from samples
#     referenced_indices: set = set()
#     for sample in samples:
#         for qa in sample.get("qa", []):
#             for ev in qa.get("evidence", []):
#                 if isinstance(ev, str) and ":" in ev and ev.startswith("D"):
#                     referenced_indices.add(ev)

#     # Map idx to (speaker, text) from corpus
#     idx_to_entry: Dict[str, Dict[str, str]] = {}
#     for entry in corpus:
#         idx = str(entry.get("idx", ""))
#         if idx in referenced_indices:
#             speaker = str(entry.get("speaker", "")).strip()
#             text = str(entry.get("text", "")).strip()
#             idx_to_entry[idx] = {"speaker": speaker, "text": text}

#     # Build docs in stable order
#     docs: List[str] = []
#     for idx in sorted(referenced_indices):
#         entry = idx_to_entry.get(idx)
#         if entry:
#             # Match original indexing shape: first line as a title-like field (speaker),
#             # second line as text content.
#             speaker = entry["speaker"]
#             text = entry["text"]
#             docs.append(f"{speaker}\n{text}")
#     return docs


# def make_docs_from_locomo10_conversations(
#     samples: List[Dict[str, Any]],
#     per_session: bool = True,
#     only_referenced: bool = True,
# ) -> List[str]:
#     """
#     Construct indexable docs directly from locomo10.json conversation fields.

#     - If per_session=True: build one document per session id (e.g., D1) per sample,
#       with first line as a title ("<sample_id>_<D#>") and following lines as
#       "speaker: text" for all turns in that session.
#     - If per_session=False: build one document per turn referenced in evidence,
#       using locomo10.json turns (title "<sample_id>_<D#>" followed by the single turn).

#     When only_referenced=True, only include sessions/turns that appear in QA evidence.
#     """
#     # Collect referenced session ids or indices from evidence
#     referenced_sessions: Dict[str, set] = {}
#     referenced_indices: Dict[str, set] = {}
#     if only_referenced:
#         for sample in samples:
#             sid = str(sample.get("sample_id", "unknown"))
#             for qa in sample.get("qa", []):
#                 for ev in qa.get("evidence", []):
#                     if isinstance(ev, str) and ev.startswith("D") and ":" in ev:
#                         doc_id, line_str = ev.split(":", 1)
#                         referenced_sessions.setdefault(sid, set()).add(doc_id)
#                         referenced_indices.setdefault(sid, set()).add(ev)

#     docs: List[str] = []
#     docs_with_ids: List[Dict[str, str]] = []
#     for sample in samples:
#         sid = str(sample.get("sample_id", "unknown"))
#         conversation = sample.get("conversation", {})
#         # Iterate session keys like "session_1", "session_2"
#         for key, session_turns in conversation.items():
#             if not isinstance(key, str) or not key.startswith("session_"):
#                 continue
#             if "date" in key:
#                 continue
            
#             session_date = conversation[key+"_date_time"]
            
#             # Determine the session id (D#) from the first turn's dia_id if available
#             session_id = None
#             if isinstance(session_turns, list) and session_turns:
#                 first_dia = str(session_turns[0].get("dia_id", ""))
#                 if first_dia and ":" in first_dia and first_dia.startswith("D"):
#                     session_id = first_dia.split(":", 1)[0]
#             if not session_id:
#                 # Fallback: use the numeric part of the key
#                 try:
#                     idx_num = key.split("_", 1)[1]
#                 except Exception:
#                     idx_num = key
#                 session_id = f"D{idx_num}"

#             # Skip sessions not referenced when only_referenced is True
#             if only_referenced and session_id not in referenced_sessions.get(sid, set()):
#                 continue

#             lines: List[str] = []
#             for turn in session_turns:
#                 spk = str(turn.get("speaker", "")).strip()
#                 txt = str(turn.get("text", "")).strip()
#                 turn_str = "DATE: " + session_date + "\n"
#                 turn_str += f"{spk} said, \"{txt}\""
#                 if "blip_caption" in turn:
#                     turn_str += ' and shared %s.' % turn["blip_caption"] + '\n'
#                 lines.append(turn_str)
#                 docs_with_ids.append({
#                     turn["dia_id"]: txt
#                 })
#             docs.extend(lines)
      
#     return docs, docs_with_ids
import sys
from typing import List, Dict, Any, Tuple

def make_docs_from_locomo10_conversations(
    samples: List[Dict[str, Any]],
    per_session: bool = True,
    only_referenced: bool = True,
    chunk_size: int = 3,
    overlap: int = 1
) -> Tuple[List[str], List[Dict[str, str]]]:

    referenced_sessions: Dict[str, set] = {}
    referenced_indices: Dict[str, set] = {}
    if only_referenced:
        for sample in samples:
            sid = str(sample.get("sample_id", "unknown"))
            for qa in sample.get("qa", []):
                for ev in qa.get("evidence", []):
                    if isinstance(ev, str) and ev.startswith("D") and ":" in ev:
                        doc_id, line_str = ev.split(":", 1)
                        referenced_sessions.setdefault(sid, set()).add(doc_id)
                        referenced_indices.setdefault(sid, set()).add(ev)

    docs: List[str] = []
    docs_with_ids: List[Dict[str, str]] = []
    for sample in samples:
        sid = str(sample.get("sample_id", "unknown"))
        conversation = sample.get("conversation", {})
        
        for key, session_turns in conversation.items():
            if not isinstance(key, str) or not key.startswith("session_"):
                continue
            if "date" in key:
                continue
            
            if not isinstance(session_turns, list) or not session_turns:
                continue
                
            session_date = conversation.get(key + "_date_time", "Unknown Date")
            
            session_id = None
            if session_turns:
                first_dia = str(session_turns[0].get("dia_id", ""))
                if first_dia and ":" in first_dia and first_dia.startswith("D"):
                    session_id = first_dia.split(":", 1)[0]
            if not session_id:
                try:
                    idx_num = key.split("_", 1)[1]
                except Exception:
                    idx_num = key
                session_id = f"D{idx_num}"

            if only_referenced and session_id not in referenced_sessions.get(sid, set()):
                continue

            for turn in session_turns:
                txt = str(turn.get("text", "")).strip()
                dia_id = turn.get("dia_id", f"{sid}_{session_id}_unknownTurn")
                docs_with_ids.append({
                    dia_id: txt
                })

            n_turns = len(session_turns)
            step = max(1, chunk_size - overlap)
            
            chunks_of_turns: List[List[Dict[str, Any]]] = []
            for i in range(0, n_turns, step):
                chunk = session_turns[i : i + chunk_size]
                if chunk:
                    chunks_of_turns.append(chunk)

            if overlap == 0 and len(chunks_of_turns) > 1 and len(chunks_of_turns[-1]) == 1:
                last_turn_list = chunks_of_turns.pop()
                chunks_of_turns[-1].extend(last_turn_list)

            for turn_chunk in chunks_of_turns:
                chunk_str_lines = [f"DATE: {session_date}"]
                
                for turn in turn_chunk:
                    spk = str(turn.get("speaker", "")).strip()
                    txt = str(turn.get("text", "")).strip()
                    turn_line = f"{spk} said, \"{txt}\""
                    if "blip_caption" in turn:
                        turn_line += ' and shared %s.' % turn["blip_caption"]
                    chunk_str_lines.append(turn_line)
                
                docs.append("\n".join(chunk_str_lines))

    return docs, docs_with_ids