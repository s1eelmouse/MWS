"""Token-level JSON constraints built on XGrammar.

The grammar is compiled once from the case contract. A fresh processor is
created for every generate() call because its matcher state is not reusable.
"""
import json
import platform

from schema import Facts


def compile_json_grammar(tokenizer, model_config, max_threads: int = 8):
    """Compile Facts JSON Schema to a token grammar using the model vocabulary."""
    import xgrammar as xgr
    vocab_size = getattr(model_config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("model config has no positive vocab_size; cannot compile JSON grammar")
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
    compiler = xgr.GrammarCompiler(tokenizer_info, max_threads=max_threads)
    # Schema declaration order matches to_answer(), so the grammar keeps the
    # field order seen during fine-tuning instead of introducing a new one.
    return compiler.compile_json_schema(
        json.dumps(Facts.model_json_schema(), ensure_ascii=False),
        any_whitespace=False, separators=(",", ":"), strict_mode=True)


class GrammarLogitsProcessor:
    """Mask logits to the compiled grammar, one matcher per batch row.

    XGrammar's stock HF processor selects Triton for CUDA. Triton has no
    Windows wheel, and its JIT needs Python headers on Linux, so the backend
    can be pinned with JSON_GRAMMAR_BACKEND. Windows CUDA always uses the
    portable torch_native kernel when the setting is auto.
    """

    def __init__(self, compiled_grammar, backend="auto"):
        self.compiled_grammar = compiled_grammar
        self.backend = backend
        self.matchers = []
        self.token_bitmask = None
        self.prefilled = False
        self.batch_size = 0
        self.prefer_torch_native = platform.system() == "Windows"

    def __call__(self, input_ids, scores):
        import xgrammar as xgr
        if not self.matchers:
            self.batch_size = input_ids.shape[0]
            self.matchers = [xgr.GrammarMatcher(self.compiled_grammar)
                             for _ in range(self.batch_size)]
            self.token_bitmask = xgr.allocate_token_bitmask(
                self.batch_size, self.compiled_grammar.tokenizer_info.vocab_size)
        if input_ids.shape[0] != self.batch_size:
            raise RuntimeError(
                f"Grammar matcher batch is {self.batch_size}, got {input_ids.shape[0]}")
        if not self.prefilled:
            self.prefilled = True
        else:
            for index in range(self.batch_size):
                if not self.matchers[index].is_terminated():
                    token = input_ids[index][-1].item()
                    if not self.matchers[index].accept_token(token):
                        raise RuntimeError("Sampled token is outside the Facts JSON grammar")
        for index in range(self.batch_size):
            if not self.matchers[index].is_terminated():
                self.matchers[index].fill_next_token_bitmask(self.token_bitmask, index)
        if scores.device.type != "cuda":
            scores_cpu = scores.to("cpu")
            xgr.apply_token_bitmask_inplace(scores_cpu, self.token_bitmask, backend="cpu")
            return scores_cpu.to(scores.device)
        backend = self.backend
        if backend == "auto" and self.prefer_torch_native:
            backend = "torch_native"
        xgr.apply_token_bitmask_inplace(
            scores, self.token_bitmask.to(scores.device), backend=backend)
        return scores
