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
    - This implementation attacks token IDs (actually subword tokens), not true words.
    - So it is more accurate to call it a token-level discrete attack than a strict PWWS word attack.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer_name: str = "opt-1.3b",
        bit_number: int = 16,
        device: str = "cuda",
        target_label: int = 0,
        sem_model_name: str = "./huggingface/sbert_model",  # [NEW] 语义模型名称
        sim_threshold: float = 0.7,                                        # [NEW] 相似度阈值，低于此值时发出警告
    ) -> None:
        super().__init__()
        self.device = device
        self.bit_number = bit_number
        self.target_label = target_label
        self.attack_logs = []
        self.sim_threshold = sim_threshold  # [NEW]

        print("[Init] Loading UPV Gradient Attacker resources...")
        self.tokenizer = AutoTokenizer.from_pretrained("./huggingface/opt-1.3b")
        #self.tokenizer = LlamaTokenizer.from_pretrained("/mnt/user-data/fsb/projects/unforgeable_watermark/models/llama-7b-hf")
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
        import torch.nn.functional as F

        embeddings = self.sem_model.encode(
            [text_a, text_b],
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=True,   # L2 归一化，点积即余弦相似度
        )
        similarity = F.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0),
        ).item()
        return similarity

    # ------------------------------------------------------------------ #

    def _int_to_bin_list(self, n: int):
        """Convert integer token id to fixed-length binary bit list."""
        bin_str = format(int(n), "b")
        if len(bin_str) > self.bit_number:
            bin_str = bin_str[-self.bit_number:]
        else:
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
        return torch.tensor(
            self._int_to_bin_list(n),
            dtype=torch.float32,
            device=self.device
        )

    def _special_token_ids(self):
        """Collect tokenizer special token ids safely."""
        special_ids = set()
        for attr in [
            "bos_token_id", "eos_token_id", "pad_token_id",
            "unk_token_id", "cls_token_id", "sep_token_id", "mask_token_id",
        ]:
            value = getattr(self.tokenizer, attr, None)
            if value is not None:
                special_ids.add(value)
        return special_ids

    def _is_reasonable_token_id(self, tok_id: int) -> bool:
        """Heuristic filter to avoid obviously bad candidates."""
        if tok_id in self._special_token_ids():
            return False
        try:
            piece = self.tokenizer.decode(
                [tok_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception:
            return False

        if piece is None:
            return False
        piece = piece.replace("\ufffd", "").strip()
        if not piece or not piece.isprintable() or len(piece) > 20:
            return False
        return True

    def _get_candidates(self, word=None, max_candidates: int = 50):
        """Get candidate replacement token ids."""
        vocab_size = self.tokenizer.vocab_size
        candidates = set()
        max_trials = max_candidates * 30
        trials = 0
        while len(candidates) < max_candidates and trials < max_trials:
            trials += 1
            rand_id = random.randint(0, vocab_size - 1)
            if self._is_reasonable_token_id(rand_id):
                candidates.add(rand_id)
        return list(candidates)

    def _decode_token_ids(self, token_ids):
        """Decode token ids to text."""
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

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

    def edit(self, text: str, reference=None):
       
        if not text:
            return text

        global GLOBAL_ATTACK_LOGS
        if "GLOBAL_ATTACK_LOGS" not in globals():
            GLOBAL_ATTACK_LOGS = []

        original_text = text         
        current_text = text
        modified_indices = set()
        original_token_ids = None

        total_queries = 0
        pred_prob = None

        max_steps = 100
        batch_size = 20
        search_k = 50
        min_loss_drop = 1e-6
        revisit_penalty = 0.35
        candidates_per_pos = 12

        for step in range(max_steps):
            features_origin, token_ids = self._text_to_bits(current_text)
            if features_origin is None or len(token_ids) == 0:
                break

            if original_token_ids is None:
                original_token_ids = token_ids.copy()

            seq_len = len(token_ids)

            features_check = features_origin.clone().detach().requires_grad_(True)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(features_check)
            total_queries += 1

            probs = outputs.squeeze()
            if probs.dim() == 0:
                probs = probs.unsqueeze(0)

            pred_prob = probs.item()
            pred_label = 1 if pred_prob > 0.5 else 0

            # Success check
            if pred_label == self.target_label:
                changed_cnt = 0
                if original_token_ids is not None and len(original_token_ids) == len(token_ids):
                    changed_cnt = sum(
                        1 for a, b in zip(original_token_ids, token_ids) if a != b
                    )
                else:
                    changed_cnt = len(modified_indices)

                rate = (changed_cnt / max(seq_len, 1)) * 100.0

                # [NEW] 计算语义相似度
                sem_sim = self.compute_semantic_similarity(original_text, current_text)
                sim_warn = sem_sim < self.sim_threshold

                print(
                    f"[Success] Attack succeeded! "
                    f"Queries: {total_queries} | Rate: {rate:.2f}% | "
                    f"Semantic Similarity: {sem_sim:.4f}"
                    + (" ⚠ Below threshold" if sim_warn else "")
                )

                log_item = {
                    "success": True,
                    "queries": total_queries,
                    "rate": rate,
                    "prob": pred_prob,
                    "steps": step + 1,
                    "semantic_similarity": sem_sim,   # [NEW]
                    "sim_below_threshold": sim_warn,  # [NEW]
                }
                self.attack_logs.append(log_item)
                GLOBAL_ATTACK_LOGS.append(log_item)
                return current_text

            # Backward
            current_loss_val, _ = self._compute_attack_loss(probs)
            current_loss_val.backward()

            bits_grad = features_check.grad
            if bits_grad is None:
                break

            grad_norm = bits_grad.norm(dim=2).squeeze(0)

            for idx in modified_indices:
                if 0 <= idx < seq_len:
                    grad_norm[idx] = grad_norm[idx] * revisit_penalty

            curr_k = min(search_k, seq_len)
            if curr_k <= 0:
                break

            _, top_grad_indices = torch.topk(grad_norm, curr_k)

            candidates_pool = []
            seen_moves = set()

            for i in range(curr_k):
                target_pos = top_grad_indices[i].item()
                target_bit_grad = bits_grad[0, target_pos]
                original_bits = features_origin[0, target_pos]
                original_id = token_ids[target_pos]

                cand_ids = self._get_candidates(None, max_candidates=candidates_per_pos)

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

            best_move = None
            best_real_drop = -float("inf")

            if candidates_pool:
                candidates_pool.sort(key=lambda x: x["score"], reverse=True)
                batch_candidates = candidates_pool[:batch_size]

                batch_input_list = []
                valid_cands = []

                for cand in batch_candidates:
                    new_feats = features_origin.clone()
                    new_feats[0, cand["pos"]] = self._id_to_bits_tensor(cand["new_id"])
                    batch_input_list.append(new_feats)
                    valid_cands.append(cand)

                if batch_input_list:
                    batch_tensor = torch.cat(batch_input_list, dim=0)

                    with torch.no_grad():
                        batch_out = self.model(batch_tensor)
                        total_queries += 1

                    batch_probs = batch_out.squeeze()
                    if batch_probs.dim() == 0:
                        batch_probs = batch_probs.unsqueeze(0)

                    curr_loss_scalar = current_loss_val.item()

                    for i, prob in enumerate(batch_probs):
                        p = prob.item()
                        prob_target = 1.0 - p if self.target_label == 0 else p
                        real_loss = -torch.log(torch.tensor(prob_target + 1e-10)).item()
                        drop = curr_loss_scalar - real_loss

                        if drop > best_real_drop:
                            best_real_drop = drop
                            best_move = valid_cands[i]

            if best_move is not None and best_real_drop > min_loss_drop:
                token_ids[best_move["pos"]] = best_move["new_id"]
                current_text = self._decode_token_ids(token_ids)
                modified_indices.add(best_move["pos"])
            else:
                kick_pos = top_grad_indices[0].item() if curr_k > 0 else -1

                if kick_pos != -1:
                    kick_candidates = self._get_candidates(None, max_candidates=20)
                    kick_candidates = [cid for cid in kick_candidates if cid != token_ids[kick_pos]]

                    if kick_candidates:
                        rand_id = random.choice(kick_candidates)
                        token_ids[kick_pos] = rand_id
                        current_text = self._decode_token_ids(token_ids)
                        modified_indices.add(kick_pos)
                    else:
                        print("[-] No valid kick candidates found. Stopping.")
                        break
                else:
                    print("[-] No valid kick position found. Stopping.")
                    break

        # Failure log
        final_features, final_token_ids = self._text_to_bits(current_text)
        if original_token_ids is not None and final_token_ids and len(original_token_ids) == len(final_token_ids):
            changed_cnt = sum(1 for a, b in zip(original_token_ids, final_token_ids) if a != b)
            denom = len(final_token_ids)
        else:
            changed_cnt = len(modified_indices)
            denom = max(len(modified_indices), 1)

        rate = (changed_cnt / max(denom, 1)) * 100.0

        # [NEW] 计算语义相似度（失败情况同样记录）
        sem_sim = self.compute_semantic_similarity(original_text, current_text)
        sim_warn = sem_sim < self.sim_threshold

        print(
            f"[-] Attack finished (not fully flipped). "
            f"Queries: {total_queries} | Rate: {rate:.2f}% | "
            f"Semantic Similarity: {sem_sim:.4f}"
            + (" ⚠ Below threshold" if sim_warn else "")
        )

        log_item = {
            "success": False,
            "queries": total_queries,
            "rate": rate,
            "prob": pred_prob,
            "steps": max_steps,
            "semantic_similarity": sem_sim,   # [NEW]
            "sim_below_threshold": sim_warn,  # [NEW]
        }
        self.attack_logs.append(log_item)
        GLOBAL_ATTACK_LOGS.append(log_item)

        return current_text

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
            "avg_rate_success":       _avg(successes, "rate"),
            "avg_sem_sim_success":    _avg(successes, "semantic_similarity"),
            # 失败样本统计
            "avg_queries_failure":    _avg(failures,  "queries"),
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

