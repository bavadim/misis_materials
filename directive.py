#!/usr/bin/env python3

import os, multiprocessing
from enum import Enum
from functools import cache
from typing import Dict, Callable, Tuple
import torch
from transformers import (
	LogitsProcessor, LogitsProcessorList,
	AutoTokenizer, AutoModelForCausalLM,
	StoppingCriteria, StoppingCriteriaList
)
from concurrent.futures import ProcessPoolExecutor
from profiling import time_avg, print_profile, reset_profile

class CheckResult(Enum):
	ACCEPT = 1
	REJECT = -1
	ABSTAIN = 0

@time_avg
@cache
def check(text: str) -> CheckResult:
	return CheckResult.ACCEPT if text.endswith("?") else CheckResult.ABSTAIN

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true") 
def _init_worker():
			os.environ["TOKENIZERS_PARALLELISM"] = "false"
	
n_proc = os.cpu_count()
pool = ProcessPoolExecutor(max_workers=n_proc, initializer=_init_worker)


class RuleLogitsProcessor(LogitsProcessor):
	def __init__(self, tokenizer, rule, top_p=0.98, bias=10.0, forbid=-1e4, n_proc: int | None = None):	
		self.tok, self.top_p = tokenizer, top_p
		self.bias, self.forbid = bias, forbid
		self._rule: Callable[[str], CheckResult] = rule

	@time_avg
	def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
		for b, seq in enumerate(input_ids):
			prefix = self.tok.decode(seq, skip_special_tokens=False)

			mask = self._nucleus_mask(scores[b])
			rule_applied, new_scores = self._apply_rules(prefix, scores[b], mask)

			if not rule_applied:
				_, new_scores = self._apply_rules(prefix, scores[b], ~mask)

		return new_scores
 
	def _nucleus_mask(self, row: torch.Tensor) -> torch.Tensor:
		probs, idx   = torch.sort(torch.softmax(row, -1), descending=True)
		cdf          = torch.cumsum(probs, 0)
		keep_sorted  = (cdf < self.top_p) 
		mask = torch.zeros_like(keep_sorted).scatter_(0, idx, keep_sorted)
		return mask 

	@time_avg
	def _apply_rules(self, prefix: str, row: torch.Tensor, cand_mask: torch.Tensor) -> Tuple[bool, torch.Tensor]:
		new_row = row.clone()

		cand_ids = torch.where(cand_mask)[0]
		piece_list = self.tok.convert_ids_to_tokens(cand_ids.tolist())
		texts  = [prefix + self.tok.convert_tokens_to_string([p]) for p in piece_list if p is not None]
		verdicts = list(pool.map(self._rule, texts, chunksize=max(1, len(texts) // n_proc)))

		accept_idx  = [i for i, v in enumerate(verdicts) if v is CheckResult.ACCEPT]
		reject_idx  = [i for i, v in enumerate(verdicts) if v is CheckResult.REJECT]

		if accept_idx:
			new_row[cand_ids[accept_idx]] += self.bias
		if reject_idx:
			new_row[cand_ids[reject_idx]]  = self.forbid

		return bool(accept_idx or reject_idx), new_row


class StreamStoppingCriteria(StoppingCriteria):
	def __init__(self, tokenizer, eos_token_id):
		self.tokenizer = tokenizer
		self.eos_token_id = eos_token_id

	@time_avg
	def __call__(self, input_ids, scores, **kwargs):
		last_token_id = input_ids[0, -1].item()
		print(self.tokenizer.decode([last_token_id], skip_special_tokens=True), end="", flush=True)
		return last_token_id == self.eos_token_id

if __name__ == "__main__":
	model_id = "Qwen/Qwen3-8B"
	tok = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").eval()

	proc = RuleLogitsProcessor(tok, check)
	text_in = "Ask me anything"
	inputs = tok(text_in, return_tensors="pt").to(model.device)

	stopping_criteria = StoppingCriteriaList([StreamStoppingCriteria(tok, tok.eos_token_id)])

	res = model.generate(
		**inputs,
		max_new_tokens=64,
		do_sample=False,                      # greedy; можно включить sampling
		logits_processor=LogitsProcessorList([proc]),
		stopping_criteria=stopping_criteria,
	)

	print_profile()
	pool.shutdown()