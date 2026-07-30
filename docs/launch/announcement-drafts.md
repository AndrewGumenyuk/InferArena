# Launch announcement drafts

Ready-to-post drafts for announcing InferArena. Personalize the bracketed
bits, then post. Recommended order: dev.to/Medium first (the long-form story),
then HN and Reddit linking to the repo, then the X thread linking to the blog.

Golden rule: lead with the **Sarathi reproduction and the honest fidelity
docs**, not with "I made a framework." Ask for criticism explicitly.

---

## 1. Hacker News

**Title:**

```
Show HN: InferArena – Reproduce LLM scheduling research as 74-line plugins
```

**Text:**

```
Every LLM scheduling paper rebuilds the same infrastructure before testing a
single idea: fork vLLM, hack the scheduler, write a benchmark harness,
reimplement the baselines.

InferArena is an open-source experimentation platform that replaces that with
one interface (schedule(state) -> Batch) and one command (inferarena compare).

To prove the workflow, I reproduced Sarathi-Serve (OSDI'24) as a 74-line
plugin. Same workload, same seed: FCFS finishes 2/64 requests, SJF 33/64,
Sarathi-Serve 64/64 — no GPU, no engine fork, runs on a laptop in seconds.

Just as important: the docs say exactly what the simulator does NOT model
(no memory limits, no preemption cost, flat decode time), and the real-engine
adapters are honestly labeled as workload replay, not in-engine execution.
There's a published calibration script + Colab notebook to validate the
simulator against real vLLM — results pending, contributions welcome.

Repo: https://github.com/AndrewGumenyuk/InferArena
Case study: https://github.com/AndrewGumenyuk/InferArena/blob/main/docs/explanation/case-study.md

Happy to answer questions — and I'd genuinely like to hear where the
abstraction breaks for people who work on serving systems.
```

---

## 2. Reddit r/LocalLLaMA

**Title:**

```
[Project] InferArena: test LLM inference scheduling ideas without forking vLLM — Sarathi-Serve reproduced as a 74-line plugin
```

**Body:**

```
I've been working on an open-source experimentation platform for LLM
inference systems and just published v0.1.0.

**The problem:** if you want to test a new scheduling/caching idea today, you
fork an inference engine, modify internals, build a custom benchmark, and
reimplement baselines — before you learn anything about your idea.

**What it does:** you implement one method (schedule(state) -> Batch), and the
framework handles workloads, execution, metrics, and reports. Six built-in
schedulers including a faithful Sarathi-Serve (arXiv:2403.02310) reproduction,
simulation on a laptop (no GPU needed), and replay adapters for
vLLM/SGLang/TensorRT-LLM.

**Case study:** under variable prompt lengths and a tight token budget, FCFS
completes 2/64 requests, SJF 33/64, Sarathi-Serve 64/64. Full analysis with
the survivorship-bias caveats:
https://github.com/AndrewGumenyuk/InferArena/blob/main/docs/explanation/case-study.md

**What it's honest about:** the simulator is analytical, not calibrated
(yet — calibration script + free Colab notebook included); plugins run in
simulation only, real-engine adapters replay workloads from the outside;
latency percentiles only cover completed requests.

GitHub: https://github.com/AndrewGumenyuk/InferArena

What would make this genuinely useful to you? Specific criticism very welcome —
especially from anyone who's modified vLLM's scheduler.
```

---

## 3. X / Twitter thread

```
1/ Every LLM scheduling paper rebuilds the same infra before testing one idea:
fork vLLM → hack scheduler → custom benchmark → reimplement baselines.

I built InferArena to replace that with one method and one command. 🧵

2/ The core abstraction is tiny:

class MyScheduler(Scheduler):
    def schedule(self, state) -> Batch: ...

Implement it once → inferarena compare → table, plot, report. No GPU needed.

3/ Proof it works: Sarathi-Serve (OSDI'24) reproduced as a 74-line plugin.

Same workload, same seed:
• FCFS: 2/64 requests completed
• SJF: 33/64
• Sarathi-Serve: 64/64

4/ Just as important: the docs say exactly what the simulator does NOT model,
and there's a published script + free Colab notebook to calibrate it against
real vLLM. Honest simulators > impressive simulators.

5/ Open source (Apache 2.0), v0.1.1:
https://github.com/AndrewGumenyuk/InferArena

Feedback welcome — especially "here's where this abstraction breaks."
```

---

## 4. dev.to / Medium

Cross-post `docs/blog/reproducing-llm-scheduling-research.md` as-is. Add at
the top: *"InferArena is open source:
https://github.com/AndrewGumenyuk/InferArena — v0.1.1, Apache 2.0."* On
dev.to, set the canonical URL if you later host it on your own blog.

---

## 5. Community outreach (after posting)

Open **discussions, not ads**, in:

- vLLM GitHub Discussions / Slack — "what would make an experimentation
  harness like this useful to engine contributors?"
- Vidur repo — fidelity comparison is a natural conversation
- SGLang Discord

One message per community, tailored, linking the calibration notebook and
asking what workloads/policies they'd want reproduced.
