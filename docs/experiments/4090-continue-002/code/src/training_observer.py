"""Append-only loss and resource journal, independent of rolling checkpoints."""
import json
import time


def make_observer(directory, torch):
    from transformers import TrainerCallback

    class Observer(TrainerCallback):
        def __init__(self):
            self.start = time.perf_counter()
            self.path = directory / "events.jsonl"
            self.path.touch(exist_ok=False)

        def on_log(self, args, state, control, logs=None, **kwargs):
            import psutil
            event = {"step": state.global_step, "epoch": state.epoch,
                     "elapsed_seconds": time.perf_counter() - self.start,
                     "metrics": logs or {}, "process_rss_bytes": psutil.Process().memory_info().rss}
            if torch.cuda.is_available():
                event.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                             cuda_reserved_bytes=torch.cuda.memory_reserved())
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    return Observer()
