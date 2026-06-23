import torch
import xgrammar
import llguidance
import llguidance.hf
import llguidance.torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.logits_process import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
)

MAX_ITEMS_IN_ROW = 100_000


# --- grammar recognizer (llguidance) -----------------------------------------


class GrammarRecognizer:
    """How far a token sequence parses, plus the tokens allowed next as a vocab
    bitmask."""

    def __init__(self, grammar_str: str, tokenizer):
        ll_grammar = llguidance.grammar_from("grammar", grammar_str)
        self.ll_tokenizer = llguidance.hf.from_tokenizer(tokenizer)
        limits = llguidance.LLParserLimits(max_items_in_row=MAX_ITEMS_IN_ROW)
        err = llguidance.LLMatcher.validate_grammar(
            ll_grammar, self.ll_tokenizer, limits=limits
        )
        if err:
            raise ValueError(f"Grammar error: {err}")
        self.ll_matcher = llguidance.LLMatcher(
            self.ll_tokenizer, ll_grammar, limits=limits
        )
        self.index = 0
        self._bitmask = llguidance.torch.allocate_token_bitmask(
            1, self.ll_tokenizer.vocab_size
        )

    def reset(self) -> None:
        self.ll_matcher.reset()
        self.index = 0

    def try_advance(self, token_ids) -> bool:
        """Feed any tokens not yet consumed; return whether they all parsed."""
        new = token_ids[self.index :].tolist()
        r = self.ll_matcher.try_consume_tokens(new)
        if (r == 0 and len(new) == 1 and new[0] == self.ll_tokenizer.eos_token
                and self.ll_matcher.is_accepting()):
            r = 1  # consume a trailing EOS manually once the matcher accepts
        self.index += r
        if self.ll_matcher.is_error():
            return False
        return r == len(new)

    def is_accepting(self) -> bool:
        return self.ll_matcher.is_accepting()

    def filter_vocab(self):
        llguidance.torch.fill_next_token_bitmask(self.ll_matcher, self._bitmask, 0)
        return self._bitmask


# --- ASAp oracle trie ---------------------------------------------------------


class TrieNode:
    """One generated-prefix position. `raw_logprob`: the model's log-softmax here;
    `log_theta`: the accumulated CARS adjustment (-inf for illegal/dead-end tokens)."""

    __slots__ = ("children", "parent", "raw_logprob", "log_theta")

    def __init__(self, parent: "TrieNode | None" = None):
        self.children: dict[int, TrieNode] = {}
        self.parent = parent
        self.raw_logprob = None
        self.log_theta = None

    def child(self, token_id: int) -> "TrieNode":
        if token_id not in self.children:
            self.children[token_id] = TrieNode(self)
        return self.children[token_id]


# --- CARS logits processor ----------------------------------------------------


class GrammarAlignedLogitsProcessor(LogitsProcessor):
    """Per-step logit adjustment for CARS (learn level 3): mask grammar-illegal
    tokens every step (zeroes only impossible tokens, so the distribution over
    valid strings is preserved) and, across generations, subtract the mass of
    dead-end prefixes. The oracle trie persists between generations (the
    "learning"); reset() rewinds the cursor without clearing it.
    """

    def __init__(self, tokenizer, grammar: GrammarRecognizer, device):
        self.tokenizer = tokenizer
        self.grammar = grammar
        self.device = device
        self.vocab_size = grammar.ll_tokenizer.vocab_size
        self.root = TrieNode()
        self.temperature = 1.0  # set per-generation; see generate()
        self.reset()

    def reset(self) -> None:
        self.grammar.reset()
        self.start_index = None
        self.generated = None
        self.node = self.root
        self.depth = 0
        self.recompute_needed = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self._set_generated(input_ids)
        is_root = len(self.generated) == 0

        # Advance the parser over the latest token; reject if illegal.
        if not self.grammar.try_advance(self.generated):
            self._fail()

        if not is_root:
            self.node = self.node.child(self.generated[-1].item())
            self.depth += 1

        if self.node.raw_logprob is None:  # first visit: record raw probs + grammar mask
            # Record log-probs at the sampling temperature so the trie's mass
            # accounting (_recompute) is in the same space as what is actually
            # sampled. Generation itself runs at T=1 (see generate()), so the
            # temperature is applied exactly once, here.
            self.node.raw_logprob = torch.log_softmax(
                scores / self.temperature, dim=-1
            ).cpu()
            self.node.log_theta = torch.zeros(1, scores.size(1))
            xgrammar.apply_token_bitmask_inplace(
                self.node.log_theta, self.grammar.filter_vocab()
            )
            self.node.log_theta[0, self.vocab_size :] = -float("inf")  # forbid ids past real vocab
            self.recompute_needed = True

        # Return temperature-scaled log-probs plus log_theta. log_theta is -inf for
        # tokens with no valid continuation (so this only drops impossible tokens)
        # and carries the learned dead-end adjustment on revisits. Because the
        # temperature is already baked into raw_logprob and generation runs at T=1,
        # log_theta is no longer divided by T downstream — the bug this fixes.
        scores = self.node.raw_logprob.to(self.device) + self.node.log_theta.to(self.device)
        scores[:, self.vocab_size :] = -float("inf")
        return scores

    def generation_ended(self, input_ids: torch.LongTensor) -> None:
        """Validate a finished generation (raises ValueError if it does not reach an
        accepting state). Call once after model.generate()."""
        self._set_generated(input_ids)
        if not self.grammar.try_advance(self.generated):
            self._fail()
        if (self.generated[-1] != self.tokenizer.eos_token_id
                and not self.grammar.is_accepting()):
            self._fail()
        if self.recompute_needed:
            self._recompute()

    def _set_generated(self, input_ids: torch.LongTensor) -> None:
        assert input_ids.size(0) == 1, "CARS samples one sequence at a time"
        if self.start_index is None:
            self.start_index = input_ids.size(1)  # end of the prompt
        self.generated = input_ids[0, self.start_index :]

    def _fail(self):
        """Mark the dead-end token impossible, propagate up the trie, reject the sample."""
        self.exclude_generated()
        raise ValueError(self.generated)

    def exclude_generated(self) -> None:
        """Subtract the finished sequence from the trie so future draws avoid it:
        forbid its final token and propagate the freed mass upward."""
        self.node.log_theta[0, self.generated[-1]] = -float("inf")
        self._recompute()

    def _recompute(self) -> None:
        node, depth = self.node, self.depth
        while depth > 0:
            mass = torch.log(torch.exp(node.raw_logprob[0] + node.log_theta[0]).sum())
            depth -= 1
            node = node.parent
            node.log_theta[0, self.generated[depth]] = mass


# --- model wrapper + sampling loop --------------------------------------------


class ConstrainedModel:
    """A HuggingFace causal LM constrained to a grammar via the CARS processor."""

    def __init__(self, model_id: str, grammar_str: str, **model_kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", **model_kwargs
        )
        self.model.eval()
        grammar = GrammarRecognizer(grammar_str, self.tokenizer)
        self.processor = GrammarAlignedLogitsProcessor(
            self.tokenizer, grammar, self.model.device
        )

    def format_prompt(self, prompt: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def generate(
        self, prompt_ids: torch.LongTensor, max_new_tokens: int, temperature: float
    ):
        """One constrained generation: returns the generated token ids, or raises
        ValueError on a rejection (left the grammar). On success the sequence is
        subtracted from the trie so it is never drawn again."""
        self.processor.reset()
        self.processor.temperature = temperature  # applied inside the processor
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,  # T is baked into the processor's log-probs, not re-applied here
            top_k=None,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        processors = LogitsProcessorList(
            [self.processor, InfNanRemoveLogitsProcessor()]
        )
        sequences = self.model.generate(
            prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            generation_config=config,
            tokenizer=self.tokenizer,
            logits_processor=processors,
        )
        self.processor.generation_ended(sequences)  # raises ValueError if rejected
        self.processor.exclude_generated()  # never draw this exact program again
        return sequences[0, prompt_ids.size(1) :]


def sample_programs(
    model: ConstrainedModel,
    prompt: str,
    n_programs: int,
    n_steps: int,
    max_new_tokens: int,
    temperature: float = 1.0,
) -> list[str]:
    """Draw up to `n_programs` distinct grammar-valid completions in at most
    `n_steps` attempts. Accepted programs are subtracted from the trie so they are
    not redrawn; `seen` is a backstop for two token sequences decoding to the same
    string. Prints accepted as `[i/n] ...`, rejected (off-grammar) as `[reject] ...`."""
    text = model.format_prompt(prompt)
    prompt_ids = model.tokenizer.encode(
        text, return_tensors="pt", add_special_tokens=False
    ).to(model.model.device)

    programs: list[str] = []
    seen: set[str] = set()
    for _ in range(n_steps):
        try:
            ids = model.generate(prompt_ids, max_new_tokens, temperature)
        except ValueError as rejection:
            # _fail raised ValueError(self.generated) on the off-grammar path.
            (rejected_ids,) = rejection.args
            rejected = model.tokenizer.decode(rejected_ids, skip_special_tokens=True)
            print(f"[reject] {rejected.strip()}", flush=True)
            continue  # the trie has learned to avoid this prefix
        program = model.tokenizer.decode(ids, skip_special_tokens=True).strip()
        if program in seen:
            continue  # same string via a different tokenization
        seen.add(program)
        programs.append(program)
        print(f"[{len(programs)}/{n_programs}] {programs[-1]}", flush=True)
        if len(programs) >= n_programs:
            break
    return programs
