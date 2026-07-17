"""Decoding grammars: `build_grammar` returns a grammar whose language is the programs
provably equivalent to the reference (egrammar.py). Branching makes the grammar depend
on the model's own output: the head grammar offers a whole-box form or the opening
`(if (op var number)`; DynamicRegionRecognizer then rebuilds each arm's grammar over
the guard-narrowed box mid-decode, so to the model it is one generation.
"""

import re

from base import benchmarks, regions
from synth import egrammar

# The `(if (op var number)` opening the head grammar can emit.
COND = re.compile(r"\(if \((<=|>=|<|>) (\w+) (-?\d+(?:\.\d+)?)\)")

NUMBER = 'number ::= "-"? [0-9]+ ("." [0-9]+)?'


def _cond_alts(operand: str) -> str:
    return " | ".join(f'"({op} " {operand} " " number ")"' for op in ("<", ">", "<=", ">="))


def head_grammar(benchmark, box, variables, runs) -> str:
    """Either a complete no-branch program (whole-box e-grammar), or the opening
    `(FPCore (v) (if (op var number)` with an arbitrary threshold, stopping at the `)`.
    The shared `(FPCore (v) ` prefix is one literal so the alternatives don't collide
    in the lexer when a token straddles that boundary."""
    vs = " ".join(variables)
    return "\n".join([
        f'root ::= "(FPCore ({vs}) " body',
        'body ::= e0 ")" | "(if " cond',
        egrammar.rules(benchmark, box, runs),
        f"cond ::= {_cond_alts('operand')}",
        "operand ::= " + " | ".join(f'"{v}"' for v in variables),
        NUMBER,
    ]) + "\n"


def arm_grammar(benchmark, box, runs, closes) -> str:
    """A leaf over `box`: leading space, then `closes` trailing `)` (0 for the
    then-arm; 2 for the else-arm, closing the `if` and the `FPCore`)."""
    suffix = "".join(' ")"' for _ in range(closes))
    return f'root ::= " " e0{suffix}\n{egrammar.rules(benchmark, box, runs)}\n'


class DynamicRegionRecognizer:
    """Grammar recognizer that swaps its own grammar mid-generation, so an `if`-program
    decodes in ONE pass: start on the head grammar; the moment it accepts
    `...(if (op var n)`, rebuild the then-arm's e-grammar over the sub-box the guard
    selects and continue; likewise then -> else. A drop-in for casa's
    XGrammarTokenRecognizer, so ASAp's oracle trie/reweighting stay intact -- the
    swap points are a deterministic function of the emitted prefix, so the trie's
    per-prefix masks stay valid."""

    def __init__(self, llm, benchmark, box, saturation):
        import xgrammar
        self._xgr = xgrammar
        self.llm = llm
        self.benchmark = benchmark
        self.box = box
        self.saturation = saturation
        self.tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(llm.tokenizer)
        self.vocab_size = self.tokenizer_info.vocab_size
        self._compiler = xgrammar.GrammarCompiler(self.tokenizer_info)
        self._bitmask = xgrammar.allocate_token_bitmask(1, self.vocab_size)
        variables = regions.variables_of(benchmarks.read_reference(benchmark))
        self.head_str = head_grammar(benchmark, box, variables, saturation)
        self._arm_cache: dict[tuple, str] = {}  # (side, op, var, thr) -> grammar str
        self.reset()

    def _matcher(self, grammar_str):
        return self._xgr.GrammarMatcher(self._compiler.compile_grammar(
            self._xgr.Grammar.from_ebnf(grammar_str)))

    def _arm_str(self, side, op, var, thr):
        key = (side, op, var, thr)
        if key not in self._arm_cache:
            guard = (op, var, thr) if side == "then" else (regions.NEGATE[op], var, thr)
            sub = regions.narrow_box(self.box, [guard]) or self.box
            self._arm_cache[key] = arm_grammar(self.benchmark, sub, self.saturation,
                                               0 if side == "then" else 2)
        return self._arm_cache[key]

    def reset(self):
        self.stage = "head"            # head -> then -> else -> done
        self.current_index = 0         # tokens consumed so far
        self.active = self._matcher(self.head_str)
        self._guard = None

    def try_advance_token_ids(self, token_ids) -> bool:
        toks = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
        # Token-by-token so a segment swap takes effect before the next token;
        # a grammar-boundary-straddling token can't be proposed (the active
        # segment's mask excludes it), matching the old llguidance behavior.
        while self.current_index < len(toks):
            if not self.active.accept_token(toks[self.current_index]):
                return False
            self.current_index += 1
            self._maybe_transition(toks)
        return True

    def _maybe_transition(self, toks) -> None:
        """At a segment boundary (active grammar completed) advance head -> then -> else."""
        while self.stage in ("head", "then") and self.active.is_completed():
            if self.stage == "head":
                text = self.llm.tokenizer.decode(
                    toks[:self.current_index], skip_special_tokens=True).strip()
                m = COND.search(text)
                if m is None:          # complete no-branch program
                    self.stage = "done"
                    break
                self._guard = (m.group(1), m.group(2), m.group(3))
                self.active = self._matcher(self._arm_str("then", *self._guard))
                self.stage = "then"
            else:
                self.active = self._matcher(self._arm_str("else", *self._guard))
                self.stage = "else"

    def is_accepting(self) -> bool:
        """Whether the whole composite grammar can end here."""
        return self.stage == "done" or (
            self.stage == "else" and self.active.is_completed())

    def filter_vocab(self):
        self.active.fill_next_token_bitmask(self._bitmask, 0)
        return self._bitmask

    def apply_token_bitmask(self, logits, bitmask) -> None:
        self._xgr.apply_token_bitmask_inplace(logits, bitmask)


def build_grammar(llm, benchmark, box, saturation):
    """casa.Grammar whose language is the programs equivalent to the reference."""
    import casa
    if box is None:
        return casa.Grammar.from_string(
            egrammar.build(benchmark, None, saturation), llm.tokenizer,
            engine="xgrammar")
    recognizer = DynamicRegionRecognizer(llm, benchmark, box, saturation)
    grammar = casa.Grammar.from_string(recognizer.head_str, llm.tokenizer,
                                       engine="xgrammar")
    grammar.recognizer = recognizer
    return grammar
