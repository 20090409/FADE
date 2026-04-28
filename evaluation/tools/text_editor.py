# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ================================================
# text_editor.py
# Description: Edit text using various techniques
# ================================================

import re
import copy
import os
import ast
import math
import difflib
import nltk
import torch
import random
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from nltk import pos_tag
from nltk.corpus import wordnet
from translate import Translator
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from utils.openai_utils import OpenAIAPI
from exceptions.exceptions import DiversityValueError
from evaluation.tools.oracle import QualityOracle
        
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForMaskedLM,AutoTokenizer,LlamaTokenizer
from watermark.upv.network_model import UPVDetector
import torch.nn.functional as F

GLOBAL_ATTACK_LOGS = []

class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text




class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
            Initialize the synonym substitution editor.

            Parameters:
                ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet')

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)
        
        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)
        
            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)

        return replaced_text




class SimpleGoal:
    """Replaces OpenAttack.ClassifierGoal"""
    def __init__(self, target, targeted=True):
        self.target = target
        self.targeted = targeted
    
    def check(self, prediction_label):
        if self.targeted:
            return prediction_label == self.target
        else:
            return prediction_label != self.target

class UPVGradientAttack(TextEditor):
    """
    White-box gradient-based attack for UPV watermark detector.

    Note:
    - Candidate replacements are derived from WordNet synonyms.
    - Replacement is still performed on token IDs (subword tokens), so this remains
      a constrained token-level discrete attack.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "opt-1.3b",
        bit_number: int = 16,
        device: str = "cuda",
        target_label: int = 0,
        sem_model_name: str = "./huggingface/sbert_model",  # [NEW] 语义模型名称
        sim_threshold: float = 0.82,                                       # [NEW] 相似度阈值，低于此值时发出警告
        speed_mode: str = "balanced",
    ) -> None:
        super().__init__()
        self.device = device
        self.bit_number = bit_number
        self.target_label = target_label
        self.attack_logs = []
        self.sim_threshold = sim_threshold  # [NEW]
        self._wordnet_cache = {}
        self._special_ids_cache = None
        self.success_verifier = None
        self._id_bits_cache = {}
        self._sem_embed_cache = OrderedDict()
        self._success_verifier_cache = {}
        self._max_sem_cache_size = 2048
        self._token_meta_cache = {}
        self._reasonable_token_cache = {}
        self._encoded_single_token_cache = {}
        self._token_candidate_cache = {}
        self._candidate_cache_build_size = 256
        self.speed_mode = speed_mode

        print("[Init] Loading UPV Gradient Attacker resources...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = UPVDetector(
            bit_number=bit_number,
            b_layers=5,
            input_dim=64,
            hidden_dim=128,
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=device),
            strict=False
        )
        self.model.to(device)
        # Keep RNN in train mode so cudnn allows backward w.r.t. input features.
        # Parameters are frozen below, so this does not update detector weights.
        self.model.train()

        for param in self.model.parameters():
            param.requires_grad = False

        # [NEW] 加载语义相似度模型
        print(f"[Init] Loading semantic similarity model: {sem_model_name} ...")
        from sentence_transformers import SentenceTransformer
        self.sem_model = SentenceTransformer(sem_model_name)
        self.sem_model.to(device)
        print("[Init] Semantic similarity model loaded.")

        try:
            nltk.data.find("corpora/wordnet")
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            print("[Init] Downloading WordNet data...")
            nltk.download("wordnet")
            nltk.download("omw-1.4")

    def set_success_verifier(self, verifier):
        """
        Set external success verifier.

        The verifier should accept `text: str` and return True when the attack
        is considered successful (i.e., watermark detector is fooled).
        """
        self.success_verifier = verifier

    def _is_success_by_verifier(self, text: str, fallback_label=None) -> bool:
        """
        Unified success check.
        Priority:
        1) external verifier (if provided)
        2) fallback detector label check
        """
        if text in self._success_verifier_cache:
            return self._success_verifier_cache[text]

        if callable(self.success_verifier):
            try:
                verdict = bool(self.success_verifier(text))
                self._success_verifier_cache[text] = verdict
                return verdict
            except Exception as exc:
                print(f"[Warn] External success verifier failed: {exc}")
                self._success_verifier_cache[text] = False
                return False

        if fallback_label is None:
            _, pred_label, _ = self._predict_from_text(text)
            verdict = pred_label == self.target_label
            self._success_verifier_cache[text] = verdict
            return verdict
        verdict = fallback_label == self.target_label
        self._success_verifier_cache[text] = verdict
        return verdict

    def _get_semantic_embedding(self, text: str):
        """Get cached normalized sentence embedding."""
        cached = self._sem_embed_cache.get(text)
        if cached is not None:
            self._sem_embed_cache.move_to_end(text)
            return cached

        embedding = self.sem_model.encode(
            text,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,
        )
        # Keep an LRU cache to cap memory.
        self._sem_embed_cache[text] = embedding
        self._sem_embed_cache.move_to_end(text)
        if len(self._sem_embed_cache) > self._max_sem_cache_size:
            self._sem_embed_cache.popitem(last=False)
        return embedding

    # ------------------------------------------------------------------ #
    # [NEW] 语义相似度计算                                                  #
    # ------------------------------------------------------------------ #

    def compute_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """
        计算两段文本之间的余弦语义相似度。

        Args:
            text_a: 原始文本
            text_b: 攻击后文本

        Returns:
            similarity: float，范围 [-1, 1]，越接近 1 语义越相似
        """
        emb_a = self._get_semantic_embedding(text_a)
        emb_b = self._get_semantic_embedding(text_b)
        return torch.dot(emb_a, emb_b).item()

    # ------------------------------------------------------------------ #

    def _int_to_bin_list(self, n: int):
        """Convert integer token id to fixed-length binary bit list."""
        bin_str = format(int(n), "b")
        # Keep this identical to UPVUtils.int_to_bin_list in watermark/upv/upv.py
        # so attack-side detector inputs are consistent with final detect_watermark().
        if len(bin_str) > self.bit_number:
            bin_str = bin_str[:self.bit_number]
        bin_str = bin_str.zfill(self.bit_number)
        return [int(b) for b in bin_str]

    def _text_to_bits(self, text: str):
        """Convert text into UPV bit features and token ids."""
        if not text:
            return None, []

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = inputs["input_ids"][0].tolist()
        if len(input_ids) == 0:
            return None, []

        features = [self._int_to_bin_list(tok_id) for tok_id in input_ids]
        features = torch.tensor(
            features,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        return features, input_ids

    def _id_to_bits_tensor(self, n: int):
        """Convert one token id to tensor bit feature."""
        cached = self._id_bits_cache.get(n)
        if cached is not None:
            return cached
        tensor = torch.tensor(
            self._int_to_bin_list(n),
            dtype=torch.float32,
            device=self.device
        )
        self._id_bits_cache[n] = tensor
        return tensor

    def _special_token_ids(self):
        """Collect tokenizer special token ids safely."""
        if self._special_ids_cache is None:
            special_ids = set()
            for attr in [
                "bos_token_id", "eos_token_id", "pad_token_id",
                "unk_token_id", "cls_token_id", "sep_token_id", "mask_token_id",
            ]:
                value = getattr(self.tokenizer, attr, None)
                if value is not None:
                    special_ids.add(value)
            self._special_ids_cache = special_ids
        return self._special_ids_cache

    def _is_reasonable_token_id(self, tok_id: int) -> bool:
        """Heuristic filter to avoid obviously bad candidates."""
        cached = self._reasonable_token_cache.get(tok_id)
        if cached is not None:
            return cached

        if tok_id in self._special_token_ids():
            self._reasonable_token_cache[tok_id] = False
            return False
        try:
            piece = self.tokenizer.decode(
                [tok_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            self._reasonable_token_cache[tok_id] = False
            return False

        if piece is None:
            self._reasonable_token_cache[tok_id] = False
            return False
        piece = piece.replace("\ufffd", "").strip()
        if not piece or not piece.isprintable() or len(piece) > 20:
            self._reasonable_token_cache[tok_id] = False
            return False
        self._reasonable_token_cache[tok_id] = True
        return True

    def _normalize_word_for_wordnet(self, piece: str) -> str:
        """Normalize decoded token piece into a WordNet-queryable word."""
        if piece is None:
            return ""
        word = piece.replace("\ufffd", "").strip()
        word = re.sub(r"^[^a-zA-Z]+|[^a-zA-Z]+$", "", word)
        word = word.lower()
        if not word or not word.isalpha():
            return ""
        return word

    def _apply_case_pattern(self, template_word: str, candidate_word: str) -> str:
        """Apply coarse casing style from source token to synonym candidate."""
        if not template_word:
            return candidate_word
        if template_word.isupper():
            return candidate_word.upper()
        if template_word.istitle():
            return candidate_word.title()
        return candidate_word

    def _pluralize_word(self, word: str) -> str:
        if not word:
            return word
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return word + "es"
        return word + "s"

    def _past_tense_word(self, word: str) -> str:
        if not word:
            return word
        if word.endswith("e"):
            return word + "d"
        if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
            return word[:-1] + "ied"
        return word + "ed"

    def _gerund_word(self, word: str) -> str:
        if not word:
            return word
        if word.endswith("ie"):
            return word[:-2] + "ying"
        if word.endswith("e") and not word.endswith("ee"):
            return word[:-1] + "ing"
        return word + "ing"

    def _candidate_surface_forms(self, source_surface: str, synonym: str):
        """
        Build grammar-preserving surface forms for a synonym.
        WordNet mostly returns lemmas; matching common inflections improves text
        quality and increases one-token candidate coverage.
        """
        source = re.sub(r"[^a-zA-Z]", "", source_surface or "")
        source_lower = source.lower()
        synonym_lower = synonym.lower()

        forms = [synonym_lower]
        if len(source_lower) >= 4:
            lemma = wordnet.morphy(source_lower) or source_lower
            source_is_inflected = lemma != source_lower
            if source_lower.endswith("ing"):
                forms.append(self._gerund_word(synonym_lower))
            elif source_lower.endswith("ied"):
                forms.append(self._past_tense_word(synonym_lower))
            elif source_lower.endswith("ed"):
                forms.append(self._past_tense_word(synonym_lower))
            elif source_lower.endswith("ies"):
                forms.append(self._pluralize_word(synonym_lower))
            elif (
                source_is_inflected
                and source_lower.endswith("s")
                and not source_lower.endswith("ss")
            ):
                forms.append(self._pluralize_word(synonym_lower))

        styled_forms = []
        seen = set()
        for form in forms:
            if not form or not form.isalpha():
                continue
            styled = self._apply_case_pattern(source_surface, form)
            key = styled.lower()
            if key in seen:
                continue
            seen.add(key)
            styled_forms.append(styled)
        return styled_forms

    def _wordnet_synonyms(self, word: str):
        """
        Get WordNet-backed replacement words sorted by lexical plausibility.
        In addition to direct synonyms, include close WordNet relations. Sentence
        similarity is checked later, so this widens the attack search without
        accepting semantically poor candidates blindly.
        """
        if not word:
            return []

        cached = self._wordnet_cache.get(word)
        if cached is not None:
            return cached

        query_words = [word]
        lemma_form = wordnet.morphy(word)
        if lemma_form and lemma_form not in query_words:
            query_words.append(lemma_form)

        synonym_scores = {}

        def _add_candidate(candidate: str, score: float):
            cand = candidate.replace("_", " ").lower().strip()
            if not cand or cand == word:
                return
            # Keep lexical replacements only; phrase-level replacement would need
            # span-aware retokenization.
            if " " in cand or "-" in cand or "'" in cand:
                return
            if not cand.isalpha():
                return
            length_penalty = abs(len(cand) - len(word))
            final_score = score - 0.08 * length_penalty
            if cand not in synonym_scores or final_score > synonym_scores[cand]:
                synonym_scores[cand] = final_score

        for query_word in query_words:
            for syn in wordnet.synsets(query_word):
                for lemma in syn.lemmas():
                    lemma_count = float(lemma.count()) if hasattr(lemma, "count") else 0.0
                    _add_candidate(lemma.name(), 3.0 + lemma_count)

                    for related in lemma.derivationally_related_forms():
                        related_count = float(related.count()) if hasattr(related, "count") else 0.0
                        _add_candidate(related.name(), 1.1 + 0.5 * related_count)

                for related_syn in (
                    syn.similar_tos()
                    + syn.also_sees()
                    + syn.verb_groups()
                    + syn.attributes()
                ):
                    for related_lemma in related_syn.lemmas():
                        related_count = float(related_lemma.count()) if hasattr(related_lemma, "count") else 0.0
                        _add_candidate(related_lemma.name(), 1.6 + 0.4 * related_count)

                for related_syn in syn.hypernyms() + syn.hyponyms():
                    for related_lemma in related_syn.lemmas():
                        related_count = float(related_lemma.count()) if hasattr(related_lemma, "count") else 0.0
                        _add_candidate(related_lemma.name(), 0.7 + 0.25 * related_count)

        synonym_list = sorted(
            synonym_scores.keys(),
            key=lambda w: (synonym_scores[w], -abs(len(w) - len(word))),
            reverse=True,
        )
        self._wordnet_cache[word] = synonym_list
        return synonym_list

    def _token_id_to_word(self, tok_id: int):
        """
        Decode token id and extract:
        - normalized lowercase word for WordNet lookup
        - whether token prefers leading space
        - original surface form (for casing transfer)
        """
        cached = self._token_meta_cache.get(tok_id)
        if cached is not None:
            return cached

        try:
            piece = self.tokenizer.decode(
                [tok_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            meta = ("", False, "")
            self._token_meta_cache[tok_id] = meta
            return meta

        if piece is None:
            meta = ("", False, "")
            self._token_meta_cache[tok_id] = meta
            return meta

        prefer_leading_space = piece.startswith(" ")
        surface = piece.replace("\ufffd", "").strip()
        meta = (self._normalize_word_for_wordnet(piece), prefer_leading_space, surface)
        self._token_meta_cache[tok_id] = meta
        return meta

    def _encode_word_to_single_token_ids(self, word: str, prefer_leading_space: bool):
        """
        Map a synonym word into tokenizer ids, keeping only single-token mappings.
        """
        cache_key = (word, bool(prefer_leading_space))
        cached = self._encoded_single_token_cache.get(cache_key)
        if cached is not None:
            return cached

        forms = [word, f" {word}"]
        if prefer_leading_space:
            forms = [f" {word}", word]

        encoded_ids = []
        seen = set()
        for form in forms:
            try:
                token_ids = self.tokenizer.encode(form, add_special_tokens=False)
            except Exception:
                continue

            if len(token_ids) != 1:
                continue

            cand_id = token_ids[0]
            if cand_id in seen:
                continue
            seen.add(cand_id)

            if self._is_reasonable_token_id(cand_id):
                encoded_ids.append(cand_id)
        self._encoded_single_token_cache[cache_key] = encoded_ids
        return encoded_ids

    def _get_candidates_for_token_id(self, tok_id: int, max_candidates: int = 50):
        """Get cached WordNet candidate ids for a source token id."""
        cached = self._token_candidate_cache.get(tok_id)
        if cached is None:
            word, prefer_leading_space, source_surface = self._token_id_to_word(tok_id)
            cached = self._get_candidates(
                word=word,
                max_candidates=max(max_candidates, self._candidate_cache_build_size),
                prefer_leading_space=prefer_leading_space,
                source_surface=source_surface,
            )
            self._token_candidate_cache[tok_id] = cached
        return cached[:max_candidates]

    def _get_candidates(self,
                        word=None,
                        max_candidates: int = 50,
                        prefer_leading_space: bool = False,
                        source_surface: str = ""):
        """Get candidate replacement token ids from WordNet synonyms."""
        if not word or len(word) <= 2:
            return []

        candidates = []
        seen = set()
        for synonym in self._wordnet_synonyms(word):
            for surface_form in self._candidate_surface_forms(source_surface, synonym):
                synonym_token_ids = self._encode_word_to_single_token_ids(
                    surface_form,
                    prefer_leading_space=prefer_leading_space,
                )
                for cand_id in synonym_token_ids:
                    if cand_id in seen:
                        continue
                    seen.add(cand_id)
                    candidates.append(cand_id)
                    if len(candidates) >= max_candidates:
                        return candidates
        return candidates

    def _token_ids_to_features(self, token_ids):
        if not token_ids:
            return None
        bit_rows = [self._id_to_bits_tensor(tok_id) for tok_id in token_ids]
        return torch.stack(bit_rows, dim=0).unsqueeze(0)

    def _decode_token_ids(self, token_ids):
        """Decode token ids to text."""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    def _target_probability(self, pred_prob: float) -> float:
        return (1.0 - pred_prob) if self.target_label == 0 else pred_prob

    def _predict_from_token_ids(self, token_ids):
        features = self._token_ids_to_features(token_ids)
        if features is None:
            return None, None, None
        with torch.no_grad():
            outputs = self.model(features)
        prob = outputs.squeeze().item()
        pred_label = 1 if prob > 0.5 else 0
        target_prob = self._target_probability(prob)
        return prob, pred_label, target_prob

    def _predict_from_text(self, text: str):
        """
        Predict via text->tokenizer->bit features path, matching final detector input flow.
        """
        features, _ = self._text_to_bits(text)
        if features is None:
            return None, None, None
        with torch.no_grad():
            outputs = self.model(features)
        prob = outputs.squeeze().item()
        pred_label = 1 if prob > 0.5 else 0
        target_prob = self._target_probability(prob)
        return prob, pred_label, target_prob

    def _batch_semantic_similarity(self, source_text: str, candidate_texts):
        if not candidate_texts:
            return []

        source_vec = self._get_semantic_embedding(source_text)
        missing_texts = []
        missing_idx = []
        candidate_vecs = [None] * len(candidate_texts)

        for i, txt in enumerate(candidate_texts):
            cached = self._sem_embed_cache.get(txt)
            if cached is not None:
                self._sem_embed_cache.move_to_end(txt)
                candidate_vecs[i] = cached
            else:
                missing_texts.append(txt)
                missing_idx.append(i)

        if missing_texts:
            new_embeds = self.sem_model.encode(
                missing_texts,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=True,
            )
            if new_embeds.dim() == 1:
                new_embeds = new_embeds.unsqueeze(0)
            for k, (i, txt) in enumerate(zip(missing_idx, missing_texts)):
                emb = new_embeds[k]
                candidate_vecs[i] = emb
                self._sem_embed_cache[txt] = emb
                self._sem_embed_cache.move_to_end(txt)
                if len(self._sem_embed_cache) > self._max_sem_cache_size:
                    self._sem_embed_cache.popitem(last=False)

        stack_vecs = torch.stack(candidate_vecs, dim=0)
        sims = torch.matmul(stack_vecs, source_vec)
        return sims.detach().cpu().tolist()

    def _changed_count(self, original_ids, current_ids):
        paired_changes = sum(1 for a, b in zip(original_ids, current_ids) if a != b)
        return paired_changes + abs(len(original_ids) - len(current_ids))

    def _evaluate_candidate_token_lists(self,
                                        original_text: str,
                                        original_token_ids,
                                        candidate_token_lists,
                                        batch_size: int = 64,
                                        semantic_topk: int = None):
        """
        Evaluate candidate token-id sequences in mini-batches.
        Returns per-candidate predictions, semantic scores, and query count.
        """
        if not candidate_token_lists:
            return [], 0

        results = []
        query_cnt = 0

        for start in range(0, len(candidate_token_lists), batch_size):
            chunk_ids = candidate_token_lists[start:start + batch_size]
            chunk_feats = [self._token_ids_to_features(ids) for ids in chunk_ids]
            if not chunk_feats:
                continue
            batch_tensor = torch.cat(chunk_feats, dim=0)

            with torch.no_grad():
                batch_out = self.model(batch_tensor)
            query_cnt += 1

            batch_probs = batch_out.squeeze()
            if batch_probs.dim() == 0:
                batch_probs = batch_probs.unsqueeze(0)

            seq_len = max(len(original_token_ids), 1)
            chunk_rows = []
            for ids, prob_tensor in zip(chunk_ids, batch_probs):
                pred_prob = prob_tensor.item()
                pred_label = 1 if pred_prob > 0.5 else 0
                target_prob = self._target_probability(pred_prob)
                changed_cnt = self._changed_count(original_token_ids, ids)
                chunk_rows.append({
                    "token_ids": ids,
                    "pred_prob": pred_prob,
                    "pred_label": pred_label,
                    "target_prob": target_prob,
                    "sem": -1.0,
                    "changed_cnt": changed_cnt,
                    "rate": (changed_cnt / seq_len) * 100.0,
                })

            if semantic_topk is None or semantic_topk >= len(chunk_rows):
                sem_indices = list(range(len(chunk_rows)))
            else:
                sem_indices = sorted(
                    range(len(chunk_rows)),
                    key=lambda i: (
                        chunk_rows[i]["pred_label"] == self.target_label,
                        chunk_rows[i]["target_prob"],
                        -chunk_rows[i]["changed_cnt"],
                    ),
                    reverse=True,
                )[:semantic_topk]

            if sem_indices:
                sem_texts = [self._decode_token_ids(chunk_rows[i]["token_ids"]) for i in sem_indices]
                sem_scores = self._batch_semantic_similarity(original_text, sem_texts)
                for i, sem_val in zip(sem_indices, sem_scores):
                    chunk_rows[i]["sem"] = sem_val

            results.extend(chunk_rows)

        return results, query_cnt

    def _collect_local_moves(self,
                             token_ids,
                             bits_grad: torch.Tensor,
                             features_origin: torch.Tensor,
                             top_positions,
                             max_candidates_per_pos: int,
                             move_limit: int):
        """Collect ranked local replacement moves around gradient-salient positions."""
        moves = []
        seen_moves = set()

        for pos in top_positions:
            original_id = token_ids[pos]
            cand_ids = self._get_candidates_for_token_id(
                tok_id=original_id,
                max_candidates=max_candidates_per_pos,
            )
            if not cand_ids:
                continue

            target_bit_grad = bits_grad[0, pos]
            original_bits = features_origin[0, pos]
            for cand_id in cand_ids:
                if cand_id == original_id:
                    continue
                move_key = (pos, cand_id)
                if move_key in seen_moves:
                    continue
                seen_moves.add(move_key)

                cand_bits = self._id_to_bits_tensor(cand_id)
                diff = cand_bits - original_bits
                projected_score = -torch.dot(target_bit_grad, diff).item()
                moves.append({
                    "pos": pos,
                    "new_id": cand_id,
                    "score": projected_score,
                })

        moves.sort(key=lambda x: x["score"], reverse=True)
        return moves[:move_limit]

    def _diverse_preselect_moves(self, moves, limit: int, per_position_cap: int = 3):
        """
        Keep high-gradient moves while spreading candidates across positions.
        A pure global top-k often spends most evaluations on one salient token,
        which hurts ASR when that token has no useful synonym.
        """
        if limit <= 0 or not moves:
            return []

        selected = []
        selected_keys = set()
        per_pos_counts = {}

        for move in moves:
            pos = move["pos"]
            if per_pos_counts.get(pos, 0) >= per_position_cap:
                continue
            key = (pos, move["new_id"])
            if key in selected_keys:
                continue
            selected.append(move)
            selected_keys.add(key)
            per_pos_counts[pos] = per_pos_counts.get(pos, 0) + 1
            if len(selected) >= limit:
                return selected

        for move in moves:
            key = (move["pos"], move["new_id"])
            if key in selected_keys:
                continue
            selected.append(move)
            selected_keys.add(key)
            if len(selected) >= limit:
                break

        return selected

    def _compute_attack_loss(self, probs: torch.Tensor, epsilon: float = 1e-10):
        """Attack objective: maximize target-label probability."""
        probs = probs.squeeze()
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        if self.target_label == 0:
            prob_target_class = 1.0 - probs
        else:
            prob_target_class = probs

        loss = -torch.log(prob_target_class + epsilon)
        return loss, prob_target_class

    def _is_better_attempt_result(self, lhs: dict, rhs: dict) -> bool:
        """Compare two attempt results under quality-first attack goals."""
        if rhs is None:
            return True
        lhs_success = bool(lhs.get("success", False))
        rhs_success = bool(rhs.get("success", False))
        if lhs_success != rhs_success:
            return lhs_success and not rhs_success

        lhs_rate = float(lhs.get("rate", float("inf")))
        rhs_rate = float(rhs.get("rate", float("inf")))
        lhs_changed = int(lhs.get("changed_cnt", 10**9))
        rhs_changed = int(rhs.get("changed_cnt", 10**9))
        lhs_sem = float(lhs.get("semantic_similarity", -1.0))
        rhs_sem = float(rhs.get("semantic_similarity", -1.0))
        lhs_target = float(lhs.get("target_prob", -1.0))
        rhs_target = float(rhs.get("target_prob", -1.0))

        if lhs_success and rhs_success:
            if lhs_changed != rhs_changed:
                return lhs_changed < rhs_changed
            if lhs_rate != rhs_rate:
                return lhs_rate < rhs_rate
            if lhs_sem != rhs_sem:
                return lhs_sem > rhs_sem
            return lhs_target > rhs_target

        if lhs_target != rhs_target:
            return lhs_target > rhs_target
        if lhs_sem != rhs_sem:
            return lhs_sem > rhs_sem
        return lhs_rate < rhs_rate

    def _short_text_for_log(self, text: str, max_chars: int = 600) -> str:
        """Compact long generated text for readable console logs."""
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ..."

    def _word_change_summary(self, original_text: str, edited_text: str, max_changes: int = 24):
        """Return a compact token-level before/after change summary."""
        orig_tokens = re.findall(r"\w+|[^\w\s]", original_text or "")
        edit_tokens = re.findall(r"\w+|[^\w\s]", edited_text or "")
        matcher = difflib.SequenceMatcher(a=orig_tokens, b=edit_tokens)

        changes = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            before = " ".join(orig_tokens[i1:i2]) if i1 != i2 else "<EMPTY>"
            after = " ".join(edit_tokens[j1:j2]) if j1 != j2 else "<EMPTY>"
            changes.append(f"{tag}: {before} -> {after}")
            if len(changes) >= max_changes:
                changes.append("...")
                break
        return changes

    def _print_text_comparison(self, original_text: str, edited_text: str):
        """Print original/adversarial text pair and a short change list."""
        print("\n===== Text Comparison =====")
        print(f"[Before] {self._short_text_for_log(original_text)}")
        print(f"[After ] {self._short_text_for_log(edited_text)}")
        changes = self._word_change_summary(original_text, edited_text)
        if changes:
            print("[Changes]")
            for change in changes:
                print(f"  - {change}")
        else:
            print("[Changes] No textual change.")
        print("===========================\n")

    def _advance_no_progress(self, no_progress_steps: int, dynamic_sem_floor: float, cfg: dict):
        """Update no-progress counters and decide whether to early-stop this attempt."""
        no_progress_steps += 1
        if (
            no_progress_steps >= cfg["patience"]
            and dynamic_sem_floor > self.sim_threshold + 1e-8
        ):
            dynamic_sem_floor = max(
                self.sim_threshold,
                dynamic_sem_floor - cfg["sem_relax_step"],
            )
            no_progress_steps = 0

        hard_patience = cfg.get("hard_stop_patience")
        should_stop = (
            hard_patience is not None
            and no_progress_steps >= hard_patience
            and dynamic_sem_floor <= self.sim_threshold + 1e-8
        )
        return no_progress_steps, dynamic_sem_floor, should_stop

    def _attack_single_attempt(self,
                               original_text: str,
                               original_token_ids,
                               cfg: dict,
                               start_token_ids=None):
        """
        One constrained attack attempt.
        Objective:
        - Push detector to target label.
        - Keep semantic similarity >= threshold.
        - Prefer fewer edits.
        """
       
        current_token_ids = (
            start_token_ids.copy()
            if start_token_ids is not None
            else original_token_ids.copy()
        )
        modified_indices = {
            idx
            for idx, (orig_id, curr_id) in enumerate(zip(original_token_ids, current_token_ids))
            if orig_id != curr_id
        }
        total_queries = 0
        dynamic_sem_floor = max(self.sim_threshold, cfg["sem_floor"])
        no_progress_steps = 0

        seq_len = max(len(original_token_ids), 1)
        best_snapshot = {
            "success": False,
            "token_ids": current_token_ids.copy(),
            "pred_prob": 1.0,
            "pred_label": 1,
            "target_prob": 0.0,
            "semantic_similarity": self.compute_semantic_similarity(
                original_text,
                self._decode_token_ids(current_token_ids)
            ),
            "step": 0,
        }

        for step in range(cfg["max_steps"]):
            features_origin = self._token_ids_to_features(current_token_ids)
            if features_origin is None:
                break

            features_check = features_origin.clone().detach().requires_grad_(True)
            self.model.zero_grad(set_to_none=True)
            outputs = self.model(features_check)
            total_queries += 1

            probs = outputs.squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)

            pred_prob = probs.item()
            pred_label = 1 if pred_prob > 0.5 else 0
            target_prob = self._target_probability(pred_prob)

            current_text = self._decode_token_ids(current_token_ids)
            current_sem = self.compute_semantic_similarity(original_text, current_text)
            changed_cnt = self._changed_count(original_token_ids, current_token_ids)
            rate = (changed_cnt / seq_len) * 100.0

            candidate_snapshot = {
                "success": False,
                "token_ids": current_token_ids.copy(),
                "pred_prob": pred_prob,
                "pred_label": pred_label,
                "target_prob": target_prob,
                "semantic_similarity": current_sem,
                "step": step + 1,
                "rate": rate,
                "changed_cnt": changed_cnt,
            }
            if self._is_better_attempt_result(candidate_snapshot, best_snapshot):
                best_snapshot = candidate_snapshot

            external_gap_mode = False
            if pred_label == self.target_label and current_sem >= self.sim_threshold:
                # Strictly confirm through text->tokenizer path used by final detector.
                verify_prob, verify_label, verify_target = self._predict_from_text(current_text)
                total_queries += 1
                if self._is_success_by_verifier(current_text, verify_label):
                    best_snapshot["success"] = True
                    best_snapshot["pred_prob"] = verify_prob
                    best_snapshot["pred_label"] = verify_label
                    best_snapshot["target_prob"] = verify_target
                    best_snapshot["queries"] = total_queries
                    best_snapshot["steps"] = step + 1
                    return best_snapshot
                external_gap_mode = True

            current_loss_val, _ = self._compute_attack_loss(probs)
            current_loss_val.backward()
            bits_grad = features_check.grad
            if bits_grad is None:
                break

            grad_norm = bits_grad.norm(dim=2).squeeze(0)
            for idx in modified_indices:
                if 0 <= idx < len(current_token_ids):
                    grad_norm[idx] = grad_norm[idx] * cfg["revisit_penalty"]

            curr_k = min(cfg["search_k"], len(current_token_ids))
            if curr_k <= 0:
                break
            _, top_grad_indices = torch.topk(grad_norm, curr_k)

            candidates_pool = []
            seen_moves = set()
            for i in range(curr_k):
                target_pos = top_grad_indices[i].item()
                original_id = current_token_ids[target_pos]
                cand_ids = self._get_candidates_for_token_id(
                    tok_id=original_id,
                    max_candidates=cfg["candidates_per_pos"],
                )
                if not cand_ids:
                    continue

                target_bit_grad = bits_grad[0, target_pos]
                original_bits = features_origin[0, target_pos]
                for cand_id in cand_ids:
                    if cand_id == original_id:
                        continue
                    move_key = (target_pos, cand_id)
                    if move_key in seen_moves:
                        continue
                    seen_moves.add(move_key)

                    cand_bits = self._id_to_bits_tensor(cand_id)
                    diff = cand_bits - original_bits
                    projected_score = -torch.dot(target_bit_grad, diff).item()
                    candidates_pool.append({
                        "pos": target_pos,
                        "new_id": cand_id,
                        "score": projected_score,
                    })

            if not candidates_pool:
                no_progress_steps, dynamic_sem_floor, should_stop = self._advance_no_progress(
                    no_progress_steps, dynamic_sem_floor, cfg
                )
                if should_stop:
                    break
                continue

            candidates_pool.sort(key=lambda x: x["score"], reverse=True)
            eval_cands = self._diverse_preselect_moves(
                candidates_pool,
                cfg["preselect_size"],
                per_position_cap=cfg.get("preselect_per_position", 3),
            )
            if not eval_cands:
                no_progress_steps, dynamic_sem_floor, should_stop = self._advance_no_progress(
                    no_progress_steps, dynamic_sem_floor, cfg
                )
                if should_stop:
                    break
                continue

            curr_loss_scalar = current_loss_val.item()
            quick_rows = []
            for batch_start in range(0, len(eval_cands), cfg["batch_size"]):
                batch_cands = eval_cands[batch_start:batch_start + cfg["batch_size"]]
                batch_tensor = features_origin.expand(len(batch_cands), -1, -1).clone()
                for i, cand in enumerate(batch_cands):
                    batch_tensor[i, cand["pos"]] = self._id_to_bits_tensor(cand["new_id"])

                with torch.no_grad():
                    batch_out = self.model(batch_tensor)
                total_queries += 1

                batch_probs = batch_out.squeeze()
                if batch_probs.dim() == 0:
                    batch_probs = batch_probs.unsqueeze(0)

                for i, prob in enumerate(batch_probs):
                    p = prob.item()
                    pred = 1 if p > 0.5 else 0
                    prob_target = self._target_probability(p)
                    real_loss = -math.log(prob_target + 1e-10)
                    drop = curr_loss_scalar - real_loss

                    cand = batch_cands[i]
                    pos = cand["pos"]
                    orig_id = original_token_ids[pos]
                    curr_id = current_token_ids[pos]
                    new_id = cand["new_id"]
                    cand_ids = current_token_ids.copy()
                    cand_ids[pos] = new_id
                    changed_delta = int(new_id != orig_id) - int(curr_id != orig_id)
                    new_changed = changed_cnt + changed_delta
                    quick_score = drop - cfg["rate_weight"] * max(changed_delta, 0)
                    quick_rows.append({
                        "pred_prob": p,
                        "pred_label": pred,
                        "prob_target": prob_target,
                        "target_prob": prob_target,
                        "drop": drop,
                        "quick_score": quick_score,
                        "sem": -1.0,
                        "pos": pos,
                        "new_id": new_id,
                        "token_ids": cand_ids,
                        "new_changed": new_changed,
                    })

            sem_budget = min(cfg.get("sem_eval_topk", len(quick_rows)), len(quick_rows))
            sem_indices = []
            sem_idx_set = set()

            target_first = sorted(
                [i for i, row in enumerate(quick_rows) if row["pred_label"] == self.target_label],
                key=lambda i: (quick_rows[i]["prob_target"], quick_rows[i]["drop"]),
                reverse=True,
            )
            for idx in target_first:
                if len(sem_indices) >= sem_budget:
                    break
                sem_indices.append(idx)
                sem_idx_set.add(idx)

            if len(sem_indices) < sem_budget:
                rest = [i for i in range(len(quick_rows)) if i not in sem_idx_set]
                rest.sort(
                    key=lambda i: (
                        quick_rows[i]["quick_score"],
                        quick_rows[i]["drop"],
                        quick_rows[i]["prob_target"],
                    ),
                    reverse=True,
                )
                need = sem_budget - len(sem_indices)
                sem_indices.extend(rest[:need])

            if sem_indices:
                sem_texts = []
                for idx in sem_indices:
                    row = quick_rows[idx]
                    sem_texts.append(self._decode_token_ids(row["token_ids"]))
                sem_scores = self._batch_semantic_similarity(original_text, sem_texts)
                for idx, sem_val in zip(sem_indices, sem_scores):
                    quick_rows[idx]["sem"] = sem_val

            single_has_success = any(
                row["sem"] >= self.sim_threshold and row["pred_label"] == self.target_label
                for row in quick_rows
            )
            single_has_progress = any(
                row["sem"] >= dynamic_sem_floor and row["drop"] > 0
                for row in quick_rows
            )

            pair_due = (
                cfg.get("pair_lookahead_top", 0) > 1
                and (external_gap_mode or not single_has_success)
                and step >= cfg.get("pair_start_step", 0)
                and (
                    external_gap_mode
                    or
                    not single_has_progress
                    or no_progress_steps > 0
                    or ((step + 1) % max(1, cfg.get("pair_interval", 4)) == 0)
                )
            )
            if pair_due:
                pair_moves = self._diverse_preselect_moves(
                    candidates_pool,
                    cfg["pair_lookahead_top"],
                    per_position_cap=cfg.get("pair_preselect_per_position", 2),
                )
                pair_lists = []
                seen_pair_states = set()
                max_pair_eval = cfg.get("pair_max_eval", 0)
                for i, mv1 in enumerate(pair_moves):
                    if len(pair_lists) >= max_pair_eval:
                        break
                    for mv2 in pair_moves[i + 1:]:
                        if len(pair_lists) >= max_pair_eval:
                            break
                        if mv1["pos"] == mv2["pos"]:
                            continue
                        cand_ids = current_token_ids.copy()
                        cand_ids[mv1["pos"]] = mv1["new_id"]
                        cand_ids[mv2["pos"]] = mv2["new_id"]
                        state_key = tuple(cand_ids)
                        if state_key in seen_pair_states:
                            continue
                        seen_pair_states.add(state_key)
                        pair_lists.append(cand_ids)

                pair_rows, pair_queries = self._evaluate_candidate_token_lists(
                    original_text=original_text,
                    original_token_ids=original_token_ids,
                    candidate_token_lists=pair_lists,
                    batch_size=cfg.get("pair_batch_size", cfg["batch_size"]),
                    semantic_topk=cfg.get("pair_sem_eval_topk"),
                )
                total_queries += pair_queries
                for row in pair_rows:
                    real_loss = -math.log(row["target_prob"] + 1e-10)
                    drop = curr_loss_scalar - real_loss
                    changed_delta = row["changed_cnt"] - changed_cnt
                    row["prob_target"] = row["target_prob"]
                    row["drop"] = drop
                    row["quick_score"] = drop - cfg["rate_weight"] * max(changed_delta, 0)
                    row["new_changed"] = row["changed_cnt"]
                    quick_rows.append(row)

            combo_has_success = any(
                row["sem"] >= self.sim_threshold and row["pred_label"] == self.target_label
                for row in quick_rows
            )
            combo_has_progress = any(
                row["sem"] >= dynamic_sem_floor and row["drop"] > 0
                for row in quick_rows
            )
            triple_due = (
                cfg.get("triple_lookahead_top", 0) > 2
                and (external_gap_mode or not combo_has_success)
                and step >= cfg.get("triple_start_step", 0)
                and (
                    external_gap_mode
                    or not combo_has_progress
                    or no_progress_steps > 0
                    or ((step + 1) % max(1, cfg.get("triple_interval", 6)) == 0)
                )
            )
            if triple_due:
                triple_moves = self._diverse_preselect_moves(
                    candidates_pool,
                    cfg["triple_lookahead_top"],
                    per_position_cap=cfg.get("triple_preselect_per_position", 2),
                )
                triple_lists = []
                seen_triple_states = set()
                max_triple_eval = cfg.get("triple_max_eval", 0)
                for i, mv1 in enumerate(triple_moves):
                    if len(triple_lists) >= max_triple_eval:
                        break
                    for j in range(i + 1, len(triple_moves)):
                        if len(triple_lists) >= max_triple_eval:
                            break
                        mv2 = triple_moves[j]
                        if mv1["pos"] == mv2["pos"]:
                            continue
                        for mv3 in triple_moves[j + 1:]:
                            if len(triple_lists) >= max_triple_eval:
                                break
                            if mv3["pos"] in (mv1["pos"], mv2["pos"]):
                                continue
                            cand_ids = current_token_ids.copy()
                            cand_ids[mv1["pos"]] = mv1["new_id"]
                            cand_ids[mv2["pos"]] = mv2["new_id"]
                            cand_ids[mv3["pos"]] = mv3["new_id"]
                            state_key = tuple(cand_ids)
                            if state_key in seen_triple_states:
                                continue
                            seen_triple_states.add(state_key)
                            triple_lists.append(cand_ids)

                triple_rows, triple_queries = self._evaluate_candidate_token_lists(
                    original_text=original_text,
                    original_token_ids=original_token_ids,
                    candidate_token_lists=triple_lists,
                    batch_size=cfg.get("triple_batch_size", cfg["batch_size"]),
                    semantic_topk=cfg.get("triple_sem_eval_topk"),
                )
                total_queries += triple_queries
                for row in triple_rows:
                    real_loss = -math.log(row["target_prob"] + 1e-10)
                    drop = curr_loss_scalar - real_loss
                    changed_delta = row["changed_cnt"] - changed_cnt
                    row["prob_target"] = row["target_prob"]
                    row["drop"] = drop
                    row["quick_score"] = drop - cfg["rate_weight"] * max(changed_delta, 0)
                    row["new_changed"] = row["changed_cnt"]
                    quick_rows.append(row)

            success_records = []
            progress_records = []
            threshold_records = []

            for row in quick_rows:
                sem_val = row["sem"]
                if sem_val < self.sim_threshold:
                    continue
                change_delta = max(row["new_changed"] - changed_cnt, 0)
                score = (
                    row["drop"]
                    + cfg["sem_weight"] * (sem_val - self.sim_threshold)
                    + cfg.get("target_prob_weight", 0.0) * row["prob_target"]
                    + cfg.get("change_bonus", 0.0) * change_delta
                    - cfg["rate_weight"] * change_delta
                )
                row["score"] = score
                if sem_val >= dynamic_sem_floor:
                    if row["pred_label"] == self.target_label:
                        success_records.append(row)
                    else:
                        progress_records.append(row)
                else:
                    threshold_records.append(row)

            selected = None
            if success_records:
                ranked_success = sorted(
                    success_records,
                    key=lambda x: (
                        x["prob_target"],
                        x["drop"],
                        x["sem"],
                        -x["new_changed"],
                    ),
                    reverse=True,
                )
                for candidate in ranked_success[:cfg.get("success_verify_topk", 1)]:
                    verify_text = self._decode_token_ids(candidate["token_ids"])
                    verify_prob, verify_label, verify_target = self._predict_from_text(verify_text)
                    total_queries += 1
                    if self._is_success_by_verifier(verify_text, verify_label):
                        final_changed = self._changed_count(original_token_ids, candidate["token_ids"])
                        return {
                            "success": True,
                            "token_ids": candidate["token_ids"],
                            "pred_prob": verify_prob,
                            "pred_label": verify_label,
                            "target_prob": verify_target,
                            "semantic_similarity": candidate["sem"],
                            "steps": step + 1,
                            "queries": total_queries,
                            "changed_cnt": final_changed,
                            "rate": (final_changed / seq_len) * 100.0,
                        }
                selected = ranked_success[0]
                selected["_verified_external_failed"] = True
            elif progress_records:
                selected = max(progress_records, key=lambda x: (x["score"], x["drop"], x["sem"]))
            elif threshold_records and no_progress_steps >= max(1, cfg["patience"] // 2):
                selected = max(threshold_records, key=lambda x: (x["score"], x["drop"], x["sem"]))

            if selected is None:
                no_progress_steps, dynamic_sem_floor, should_stop = self._advance_no_progress(
                    no_progress_steps, dynamic_sem_floor, cfg
                )
                if should_stop:
                    break
                continue

            current_token_ids = selected["token_ids"].copy()
            modified_indices = {
                idx
                for idx, (orig_id, curr_id) in enumerate(zip(original_token_ids, current_token_ids))
                if orig_id != curr_id
            }
            no_progress_steps = max(no_progress_steps, 1) if selected.get("_verified_external_failed", False) else 0

            if (
                selected["pred_label"] == self.target_label
                and selected["sem"] >= self.sim_threshold
                and not selected.get("_verified_external_failed", False)
            ):
                verify_text = self._decode_token_ids(current_token_ids)
                verify_prob, verify_label, verify_target = self._predict_from_text(verify_text)
                total_queries += 1
                if self._is_success_by_verifier(verify_text, verify_label):
                    final_changed = self._changed_count(original_token_ids, current_token_ids)
                    return {
                        "success": True,
                        "token_ids": current_token_ids,
                        "pred_prob": verify_prob,
                        "pred_label": verify_label,
                        "target_prob": verify_target,
                        "semantic_similarity": selected["sem"],
                        "steps": step + 1,
                        "queries": total_queries,
                        "changed_cnt": final_changed,
                        "rate": (final_changed / seq_len) * 100.0,
                    }

        # Attempt failed, return the best reachable state.
        best_ids = best_snapshot["token_ids"]
        best_changed = self._changed_count(original_token_ids, best_ids)
        return {
            "success": False,
            "token_ids": best_ids,
            "pred_prob": best_snapshot["pred_prob"],
            "pred_label": best_snapshot["pred_label"],
            "target_prob": best_snapshot["target_prob"],
            "semantic_similarity": best_snapshot["semantic_similarity"],
            "steps": cfg["max_steps"],
            "queries": total_queries,
            "changed_cnt": best_changed,
            "rate": (best_changed / seq_len) * 100.0,
        }

    def _prune_successful_result(self, original_text: str, original_token_ids, adv_token_ids):
        """
        Greedy pruning:
        revert edits one by one while preserving attack success and semantic threshold.
        This directly reduces rate and often further improves semantic similarity.
        """
        current_ids = adv_token_ids.copy()
        total_queries = 0

        improved = True
        while improved:
            improved = False
            changed_positions = [
                i for i, (a, b) in enumerate(zip(original_token_ids, current_ids)) if a != b
            ]
            for pos in changed_positions:
                trial_ids = current_ids.copy()
                trial_ids[pos] = original_token_ids[pos]

                trial_text = self._decode_token_ids(trial_ids)
                trial_prob, trial_label, _ = self._predict_from_text(trial_text)
                total_queries += 1
                if not self._is_success_by_verifier(trial_text, trial_label):
                    continue

                trial_sem = self.compute_semantic_similarity(original_text, trial_text)
                if trial_sem < self.sim_threshold:
                    continue

                current_ids = trial_ids
                improved = True

        final_text = self._decode_token_ids(current_ids)
        final_prob, final_label, _ = self._predict_from_text(final_text)
        total_queries += 1
        if not self._is_success_by_verifier(final_text, final_label):
            final_label = 1 - self.target_label
        final_sem = self.compute_semantic_similarity(original_text, final_text)
        return current_ids, final_prob, final_label, final_sem, total_queries

    def _polish_successful_result(self,
                                  original_text: str,
                                  original_token_ids,
                                  adv_token_ids,
                                  cfg: dict):
        """
        Improve text quality after success by swapping edited positions to more
        semantically similar alternatives while preserving success.
        """
        current_ids = adv_token_ids.copy()
        total_queries = 0
        current_text = self._decode_token_ids(current_ids)
        current_prob, current_label, _ = self._predict_from_text(current_text)
        total_queries += 1
        if not self._is_success_by_verifier(current_text, current_label):
            current_label = 1 - self.target_label
            return current_ids, current_prob, current_label, 0.0, total_queries

        current_sem = self.compute_semantic_similarity(original_text, current_text)
        if current_sem < self.sim_threshold:
            return current_ids, current_prob, current_label, current_sem, total_queries

        for _ in range(cfg["max_passes"]):
            improved = False
            changed_positions = [
                i for i, (a, b) in enumerate(zip(original_token_ids, current_ids)) if a != b
            ]
            if not changed_positions:
                break

            for pos in changed_positions:
                candidate_ids = [original_token_ids[pos]]
                candidate_ids.extend(
                    self._get_candidates_for_token_id(
                        original_token_ids[pos],
                        max_candidates=cfg["candidates_per_pos"],
                    )
                )

                trial_lists = []
                seen = set()
                for cand_id in candidate_ids:
                    if cand_id == current_ids[pos]:
                        continue
                    trial_ids = current_ids.copy()
                    trial_ids[pos] = cand_id
                    state_key = tuple(trial_ids)
                    if state_key in seen:
                        continue
                    seen.add(state_key)
                    trial_lists.append(trial_ids)

                if not trial_lists:
                    continue

                rows, eval_queries = self._evaluate_candidate_token_lists(
                    original_text=original_text,
                    original_token_ids=original_token_ids,
                    candidate_token_lists=trial_lists,
                    batch_size=cfg["eval_batch_size"],
                    semantic_topk=None,
                )
                total_queries += eval_queries

                rows = [
                    row for row in rows
                    if row["sem"] >= self.sim_threshold
                    and row["pred_label"] == self.target_label
                    and (
                        row["changed_cnt"] < self._changed_count(original_token_ids, current_ids)
                        or row["sem"] > current_sem + cfg["min_sem_gain"]
                    )
                ]
                rows.sort(
                    key=lambda x: (x["changed_cnt"], -x["sem"], -x["target_prob"])
                )

                for row in rows:
                    trial_text = self._decode_token_ids(row["token_ids"])
                    verify_prob, verify_label, _ = self._predict_from_text(trial_text)
                    total_queries += 1
                    if not self._is_success_by_verifier(trial_text, verify_label):
                        continue
                    current_ids = row["token_ids"]
                    current_prob = verify_prob
                    current_label = verify_label
                    current_sem = row["sem"]
                    improved = True
                    break

            if not improved:
                break

        final_text = self._decode_token_ids(current_ids)
        final_prob, final_label, _ = self._predict_from_text(final_text)
        total_queries += 1
        if not self._is_success_by_verifier(final_text, final_label):
            final_label = 1 - self.target_label
        final_sem = self.compute_semantic_similarity(original_text, final_text)
        return current_ids, final_prob, final_label, final_sem, total_queries

    def edit(self, text: str, reference=None):
        if not text:
            return text

        global GLOBAL_ATTACK_LOGS
        if "GLOBAL_ATTACK_LOGS" not in globals():
            GLOBAL_ATTACK_LOGS = []

        # Per-sample verifier cache to avoid repeated external detector calls.
        self._success_verifier_cache = {}

        _, original_token_ids = self._text_to_bits(text)
        if not original_token_ids:
            return text

        original_text = text
        seq_len = max(len(original_token_ids), 1)
        total_queries = 0

        # Multi-stage schedule with speed profiles.
        # fast:   lower latency, lower query count
        # balanced: higher ASR but slower
        if self.speed_mode == "balanced":
            attempt_cfgs = [
                {
                    "max_steps": 110,
                    "search_k": 72,
                    "candidates_per_pos": 28,
                    "preselect_size": 160,
                    "preselect_per_position": 3,
                    "batch_size": 160,
                    "revisit_penalty": 0.30,
                    "sem_floor": max(self.sim_threshold, 0.86),
                    "sem_weight": 0.34,
                    "rate_weight": 0.05,
                    "target_prob_weight": 0.08,
                    "change_bonus": 0.01,
                    "patience": 6,
                    "sem_relax_step": 0.02,
                    "sem_eval_topk": 48,
                    "hard_stop_patience": 22,
                    "pair_start_step": 2,
                    "pair_interval": 3,
                    "pair_lookahead_top": 24,
                    "pair_preselect_per_position": 2,
                    "pair_max_eval": 80,
                    "pair_batch_size": 160,
                    "pair_sem_eval_topk": 48,
                    "triple_start_step": 4,
                    "triple_interval": 4,
                    "triple_lookahead_top": 16,
                    "triple_preselect_per_position": 2,
                    "triple_max_eval": 48,
                    "triple_batch_size": 160,
                    "triple_sem_eval_topk": 48,
                    "success_verify_topk": 3,
                },
                {
                    "max_steps": 180,
                    "search_k": 144,
                    "candidates_per_pos": 40,
                    "preselect_size": 256,
                    "preselect_per_position": 4,
                    "batch_size": 256,
                    "revisit_penalty": 0.35,
                    "sem_floor": max(self.sim_threshold, 0.80),
                    "sem_weight": 0.28,
                    "rate_weight": 0.03,
                    "target_prob_weight": 0.10,
                    "change_bonus": 0.015,
                    "patience": 8,
                    "sem_relax_step": 0.02,
                    "sem_eval_topk": 72,
                    "hard_stop_patience": 28,
                    "pair_start_step": 1,
                    "pair_interval": 2,
                    "pair_lookahead_top": 36,
                    "pair_preselect_per_position": 2,
                    "pair_max_eval": 128,
                    "pair_batch_size": 256,
                    "pair_sem_eval_topk": 72,
                    "triple_start_step": 2,
                    "triple_interval": 3,
                    "triple_lookahead_top": 24,
                    "triple_preselect_per_position": 2,
                    "triple_max_eval": 80,
                    "triple_batch_size": 256,
                    "triple_sem_eval_topk": 72,
                    "success_verify_topk": 4,
                    "retry_from_original": True,
                },
                {
                    "max_steps": 280,
                    "search_k": 224,
                    "candidates_per_pos": 56,
                    "preselect_size": 320,
                    "preselect_per_position": 4,
                    "batch_size": 320,
                    "revisit_penalty": 0.40,
                    "sem_floor": self.sim_threshold,
                    "sem_weight": 0.22,
                    "rate_weight": 0.015,
                    "target_prob_weight": 0.12,
                    "change_bonus": 0.02,
                    "patience": 12,
                    "sem_relax_step": 0.01,
                    "sem_eval_topk": 96,
                    "hard_stop_patience": 36,
                    "pair_start_step": 0,
                    "pair_interval": 2,
                    "pair_lookahead_top": 48,
                    "pair_preselect_per_position": 3,
                    "pair_max_eval": 192,
                    "pair_batch_size": 320,
                    "pair_sem_eval_topk": 96,
                    "triple_start_step": 1,
                    "triple_interval": 2,
                    "triple_lookahead_top": 32,
                    "triple_preselect_per_position": 2,
                    "triple_max_eval": 128,
                    "triple_batch_size": 320,
                    "triple_sem_eval_topk": 96,
                    "success_verify_topk": 5,
                    "retry_from_original": True,
                },
                {
                    "max_steps": 340,
                    "search_k": 320,
                    "candidates_per_pos": 80,
                    "preselect_size": 384,
                    "preselect_per_position": 5,
                    "batch_size": 384,
                    "revisit_penalty": 0.45,
                    "sem_floor": self.sim_threshold,
                    "sem_weight": 0.18,
                    "rate_weight": 0.005,
                    "target_prob_weight": 0.14,
                    "change_bonus": 0.025,
                    "patience": 16,
                    "sem_relax_step": 0.005,
                    "sem_eval_topk": 128,
                    "hard_stop_patience": 52,
                    "pair_start_step": 0,
                    "pair_interval": 1,
                    "pair_lookahead_top": 60,
                    "pair_preselect_per_position": 3,
                    "pair_max_eval": 256,
                    "pair_batch_size": 384,
                    "pair_sem_eval_topk": 128,
                    "triple_start_step": 0,
                    "triple_interval": 2,
                    "triple_lookahead_top": 40,
                    "triple_preselect_per_position": 2,
                    "triple_max_eval": 192,
                    "triple_batch_size": 384,
                    "triple_sem_eval_topk": 128,
                    "success_verify_topk": 6,
                    "retry_from_original": True,
                },
                
            ]
            polish_cfg = {
                "max_passes": 2,
                "candidates_per_pos": 128,
                "eval_batch_size": 160,
                "min_sem_gain": 1e-4,
            }
            
        else:
            # Default fast profile.
            attempt_cfgs = [
                {
                    "max_steps": 64,
                    "search_k": 48,
                    "candidates_per_pos": 16,
                    "preselect_size": 80,
                    "preselect_per_position": 3,
                    "batch_size": 96,
                    "revisit_penalty": 0.30,
                    "sem_floor": max(self.sim_threshold, 0.84),
                    "sem_weight": 0.30,
                    "rate_weight": 0.05,
                    "target_prob_weight": 0.06,
                    "change_bonus": 0.01,
                    "patience": 5,
                    "sem_relax_step": 0.02,
                    "sem_eval_topk": 24,
                    "hard_stop_patience": 14,
                    "pair_start_step": 3,
                    "pair_interval": 4,
                    "pair_lookahead_top": 16,
                    "pair_preselect_per_position": 2,
                    "pair_max_eval": 48,
                    "pair_batch_size": 96,
                    "pair_sem_eval_topk": 24,
                    "triple_start_step": 6,
                    "triple_interval": 5,
                    "triple_lookahead_top": 12,
                    "triple_preselect_per_position": 2,
                    "triple_max_eval": 24,
                    "triple_batch_size": 96,
                    "triple_sem_eval_topk": 24,
                    "success_verify_topk": 2,
                },
                {
                    "max_steps": 110,
                    "search_k": 84,
                    "candidates_per_pos": 24,
                    "preselect_size": 120,
                    "preselect_per_position": 3,
                    "batch_size": 128,
                    "revisit_penalty": 0.35,
                    "sem_floor": max(self.sim_threshold, 0.78),
                    "sem_weight": 0.26,
                    "rate_weight": 0.03,
                    "target_prob_weight": 0.08,
                    "change_bonus": 0.015,
                    "patience": 6,
                    "sem_relax_step": 0.02,
                    "sem_eval_topk": 40,
                    "hard_stop_patience": 18,
                    "pair_start_step": 2,
                    "pair_interval": 3,
                    "pair_lookahead_top": 28,
                    "pair_preselect_per_position": 2,
                    "pair_max_eval": 96,
                    "pair_batch_size": 128,
                    "pair_sem_eval_topk": 48,
                    "triple_start_step": 3,
                    "triple_interval": 4,
                    "triple_lookahead_top": 20,
                    "triple_preselect_per_position": 2,
                    "triple_max_eval": 48,
                    "triple_batch_size": 128,
                    "triple_sem_eval_topk": 48,
                    "success_verify_topk": 3,
                },
                {
                    "max_steps": 160,
                    "search_k": 128,
                    "candidates_per_pos": 32,
                    "preselect_size": 160,
                    "preselect_per_position": 4,
                    "batch_size": 160,
                    "revisit_penalty": 0.40,
                    "sem_floor": self.sim_threshold,
                    "sem_weight": 0.22,
                    "rate_weight": 0.015,
                    "target_prob_weight": 0.10,
                    "change_bonus": 0.02,
                    "patience": 8,
                    "sem_relax_step": 0.01,
                    "sem_eval_topk": 56,
                    "hard_stop_patience": 24,
                    "pair_start_step": 1,
                    "pair_interval": 2,
                    "pair_lookahead_top": 40,
                    "pair_preselect_per_position": 2,
                    "pair_max_eval": 144,
                    "pair_batch_size": 160,
                    "pair_sem_eval_topk": 64,
                    "triple_start_step": 2,
                    "triple_interval": 3,
                    "triple_lookahead_top": 28,
                    "triple_preselect_per_position": 2,
                    "triple_max_eval": 96,
                    "triple_batch_size": 160,
                    "triple_sem_eval_topk": 64,
                    "success_verify_topk": 4,
                    "retry_from_original": True,
                },
            ]
            polish_cfg = {
                "max_passes": 1,
                "candidates_per_pos": 80,
                "eval_batch_size": 96,
                "min_sem_gain": 1e-4,
            }

        best_result = None
        success_result = None
        start_token_ids = None
        for cfg in attempt_cfgs:
            attempt_starts = [start_token_ids]
            if (
                cfg.get("retry_from_original", False)
                and start_token_ids is not None
                and tuple(start_token_ids) != tuple(original_token_ids)
            ):
                attempt_starts.append(None)

            seen_starts = set()
            for attempt_start in attempt_starts:
                start_key = tuple(attempt_start) if attempt_start is not None else tuple(original_token_ids)
                if start_key in seen_starts:
                    continue
                seen_starts.add(start_key)

                result = self._attack_single_attempt(
                    original_text,
                    original_token_ids,
                    cfg,
                    start_token_ids=attempt_start,
                )
                total_queries += result["queries"]
                if self._is_better_attempt_result(result, best_result):
                    best_result = result
                if result["success"]:
                    success_result = result
                    break

            if success_result is not None:
                break
            start_token_ids = best_result["token_ids"] if best_result is not None else None

        if success_result is not None:
            search_queries = total_queries
            postprocess_queries = 0

            pruned_ids, pruned_prob, pruned_label, pruned_sem, prune_queries = self._prune_successful_result(
                original_text, original_token_ids, success_result["token_ids"]
            )
            total_queries += prune_queries
            postprocess_queries += prune_queries

            if pruned_label == self.target_label and pruned_sem >= self.sim_threshold:
                final_token_ids = pruned_ids
                pred_prob = pruned_prob
                sem_sim = pruned_sem
            else:
                final_token_ids = success_result["token_ids"]
                pred_prob = success_result["pred_prob"]
                sem_sim = success_result["semantic_similarity"]

            polished_ids, polished_prob, polished_label, polished_sem, polish_queries = self._polish_successful_result(
                original_text, original_token_ids, final_token_ids, polish_cfg
            )
            total_queries += polish_queries
            postprocess_queries += polish_queries
            if polished_label == self.target_label and polished_sem >= self.sim_threshold:
                final_token_ids = polished_ids
                pred_prob = polished_prob
                sem_sim = polished_sem

                # A semantic-preserving replacement can make more edits removable.
                repruned_ids, repruned_prob, repruned_label, repruned_sem, reprune_queries = self._prune_successful_result(
                    original_text, original_token_ids, final_token_ids
                )
                total_queries += reprune_queries
                postprocess_queries += reprune_queries
                if repruned_label == self.target_label and repruned_sem >= self.sim_threshold:
                    final_token_ids = repruned_ids
                    pred_prob = repruned_prob
                    sem_sim = repruned_sem

            changed_cnt = self._changed_count(original_token_ids, final_token_ids)
            rate = (changed_cnt / seq_len) * 100.0
            sim_warn = sem_sim < self.sim_threshold
            final_text = self._decode_token_ids(final_token_ids)
            final_verify_prob, final_verify_label, _ = self._predict_from_text(final_text)
            total_queries += 1
            postprocess_queries += 1

            if not self._is_success_by_verifier(final_text, final_verify_label):
                fallback_ids = success_result["token_ids"]
                fallback_text = self._decode_token_ids(fallback_ids)
                fallback_prob, fallback_label, _ = self._predict_from_text(fallback_text)
                total_queries += 1
                postprocess_queries += 1
                if self._is_success_by_verifier(fallback_text, fallback_label):
                    final_token_ids = fallback_ids
                    final_text = fallback_text
                    pred_prob = fallback_prob
                    sem_sim = success_result["semantic_similarity"]
                    changed_cnt = self._changed_count(original_token_ids, final_token_ids)
                    rate = (changed_cnt / seq_len) * 100.0
                    sim_warn = sem_sim < self.sim_threshold
                    print(
                        f"[Success] Attack succeeded! "
                        f"Queries: {search_queries} | Postprocess Queries: {postprocess_queries} | "
                        f"Total Forwards: {total_queries} | Rate: {rate:.2f}% | "
                        f"Semantic Similarity: {sem_sim:.4f} | Postprocess fallback"
                        + (" ⚠ Below threshold" if sim_warn else "")
                    )
                    log_item = {
                        "success": True,
                        "queries": search_queries,
                        "postprocess_queries": postprocess_queries,
                        "total_queries": total_queries,
                        "query_count_mode": "batched_detector_forward_search_only",
                        "rate": rate,
                        "prob": pred_prob,
                        "steps": success_result["steps"],
                        "semantic_similarity": sem_sim,
                        "sim_below_threshold": sim_warn,
                        "postprocess_fallback": True,
                        "original_text": original_text,
                        "edited_text": final_text,
                        "word_changes": self._word_change_summary(original_text, final_text),
                    }
                    self.attack_logs.append(log_item)
                    GLOBAL_ATTACK_LOGS.append(log_item)
                    return final_text

                print(
                    f"[-] Attack finished (not fully flipped). "
                    f"Queries: {search_queries} | Postprocess Queries: {postprocess_queries} | "
                    f"Total Forwards: {total_queries} | Rate: {rate:.2f}% | "
                    f"Semantic Similarity: {sem_sim:.4f}"
                    + (" ⚠ Below threshold" if sim_warn else "")
                )
                # self._print_text_comparison(original_text, final_text)
                log_item = {
                    "success": False,
                    "queries": search_queries,
                    "postprocess_queries": postprocess_queries,
                    "total_queries": total_queries,
                    "query_count_mode": "batched_detector_forward_search_only",
                    "rate": rate,
                    "prob": final_verify_prob,
                    "steps": success_result["steps"],
                    "semantic_similarity": sem_sim,
                    "sim_below_threshold": sim_warn,
                    "original_text": original_text,
                    "edited_text": final_text,
                    "word_changes": self._word_change_summary(original_text, final_text),
                }
                self.attack_logs.append(log_item)
                GLOBAL_ATTACK_LOGS.append(log_item)
                return final_text

            print(
                f"[Success] Attack succeeded! "
                f"Queries: {search_queries} | Postprocess Queries: {postprocess_queries} | "
                f"Total Forwards: {total_queries} | Rate: {rate:.2f}% | "
                f"Semantic Similarity: {sem_sim:.4f}"
                + (" ⚠ Below threshold" if sim_warn else "")
            )
            # self._print_text_comparison(original_text, final_text)

            log_item = {
                "success": True,
                "queries": search_queries,
                "postprocess_queries": postprocess_queries,
                "total_queries": total_queries,
                "query_count_mode": "batched_detector_forward_search_only",
                "rate": rate,
                "prob": pred_prob,
                "steps": success_result["steps"],
                "semantic_similarity": sem_sim,
                "sim_below_threshold": sim_warn,
                "postprocess_fallback": False,
                "original_text": original_text,
                "edited_text": final_text,
                "word_changes": self._word_change_summary(original_text, final_text),
            }
            self.attack_logs.append(log_item)
            GLOBAL_ATTACK_LOGS.append(log_item)
            return final_text

        # Failure: return best reachable quality-preserving state.
        final_token_ids = best_result["token_ids"] if best_result is not None else original_token_ids
        final_text = self._decode_token_ids(final_token_ids)
        sem_sim = self.compute_semantic_similarity(original_text, final_text)
        changed_cnt = self._changed_count(original_token_ids, final_token_ids)
        rate = (changed_cnt / seq_len) * 100.0
        pred_prob = best_result["pred_prob"] if best_result is not None else None
        steps = best_result["steps"] if best_result is not None else 0
        sim_warn = sem_sim < self.sim_threshold

        print(
            f"[-] Attack finished (not fully flipped). "
            f"Queries: {total_queries} | Postprocess Queries: 0 | "
            f"Total Forwards: {total_queries} | Rate: {rate:.2f}% | "
            f"Semantic Similarity: {sem_sim:.4f}"
            + (" ⚠ Below threshold" if sim_warn else "")
        )
        print(
            "[-] Note: 100% ASR cannot be theoretically guaranteed for every sample; "
            "this version maximizes ASR while enforcing semantic constraints."
        )
        # self._print_text_comparison(original_text, final_text)

        log_item = {
            "success": False,
            "queries": total_queries,
            "postprocess_queries": 0,
            "total_queries": total_queries,
            "query_count_mode": "batched_detector_forward_search_only",
            "rate": rate,
            "prob": pred_prob,
            "steps": steps,
            "semantic_similarity": sem_sim,
            "sim_below_threshold": sim_warn,
            "original_text": original_text,
            "edited_text": final_text,
            "word_changes": self._word_change_summary(original_text, final_text),
        }
        self.attack_logs.append(log_item)
        GLOBAL_ATTACK_LOGS.append(log_item)
        return final_text

    # ------------------------------------------------------------------ #
    # [NEW] 汇总统计                                                        #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """
        汇总所有已记录攻击的统计信息，包括语义相似度。

        Returns:
            dict，包含成功率、平均查询次数、平均修改率、平均语义相似度等。
        """
        if not self.attack_logs:
            return {}

        total = len(self.attack_logs)
        successes = [r for r in self.attack_logs if r["success"]]
        failures  = [r for r in self.attack_logs if not r["success"]]

        def _avg(lst, key):
            vals = [r[key] for r in lst if key in r]
            return sum(vals) / len(vals) if vals else float("nan")

        stats = {
            "total_attacks":          total,
            "success_count":          len(successes),
            "failure_count":          len(failures),
            "success_rate":           len(successes) / total,
            # 成功样本统计
            "avg_queries_success":    _avg(successes, "queries"),
            "avg_postprocess_queries_success": _avg(successes, "postprocess_queries"),
            "avg_total_forwards_success": _avg(successes, "total_queries"),
            "avg_rate_success":       _avg(successes, "rate"),
            "avg_sem_sim_success":    _avg(successes, "semantic_similarity"),
            # 失败样本统计
            "avg_queries_failure":    _avg(failures,  "queries"),
            "avg_total_forwards_failure": _avg(failures, "total_queries"),
            "avg_rate_failure":       _avg(failures,  "rate"),
            "avg_sem_sim_failure":    _avg(failures,  "semantic_similarity"),
            # 全体
            "avg_sem_sim_overall":    _avg(self.attack_logs, "semantic_similarity"),
            "sim_below_threshold_cnt": sum(
                1 for r in self.attack_logs if r.get("sim_below_threshold", False)
            ),
        }

        print("\n===== Attack Summary =====")
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("==========================\n")

        return stats
