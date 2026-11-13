"""One markdown table row for tools/thread_sweep.sh from a run's summary.json."""
import json, sys

s = json.load(open(sys.argv[1]))
print(f"| {sys.argv[2]} | {str(s.get('threads_note', ''))[:40]} | {s['mean_ms']} | {s['median_ms']} | "
      f"{s['p95_ms']} | {s['fps_infer']} | {s.get('throttled_during_run')} |")
