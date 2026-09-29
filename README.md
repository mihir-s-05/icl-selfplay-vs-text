# Self-play vs. text pretraining: which one teaches in-context learning?

I read [Self-Play Pretraining with Zero Data](https://arxiv.org/abs/2609.30063) and kept coming back to one thing. The paper shows that a transformer trained only on the outputs of self-generated programs picks up in-context learning (ICL). But it never compares that ICL to what you get from ordinary pretraining on text. Is self-play actually good at teaching ICL, or does every pretraining run get there anyway?

So I ran the comparison. I used the paper's own released self-play checkpoints (100k to 24M parameters, 4 seeds each). I trained the same architecture on DCLM web text for exactly the same number of learner tokens, then tested both on the same ICL tasks.

![ICL vs model size](writeup/figs/fig1_scaling.svg)

## What I found

They learn different kinds of ICL, and the difference grows with model size.

- **Self-play is better at procedural tasks:** copying by position, tracking a stack, comparing letters. It leads by about 0.25 at 6M–24M.
- **Text is better at lookup and meaning:** recalling a key's value, or categorising a word it hasn't seen in the examples. It leads by 0.19–0.28 from 1M up, and wins even plain lookup, which needs no world knowledge.
- **The paper's raw-byte ICL format is unfair to text models.** They score about 0.02 even on "copy the first byte", so I added printable-text and word versions of the tasks.
- **Starting text training from self-play** gets word-level ICL sooner: about 2× fewer text tokens at 24M, and roughly 6–13× at 1M–3M.

Caveats: 24M parameters at most, 1–2 text seeds per size, and learner tokens matched rather than total compute. The full write-up is in [`writeup/WRITEUP.md`](writeup/WRITEUP.md).

Checkpoints: [rihim/icl-selfplay-vs-text-checkpoints](https://huggingface.co/rihim/icl-selfplay-vs-text-checkpoints). Results and logs: [rihim/icl-selfplay-vs-text-results](https://huggingface.co/datasets/rihim/icl-selfplay-vs-text-results).

## Running it

```bash
git clone https://github.com/nourya-aliz/self_play_pretraining   # the model class comes from here
git -C self_play_pretraining apply ../patches/self_play_pretraining.patch
pip install -r requirements.txt

# rebuild the figures and tables from the released results
hf download rihim/icl-selfplay-vs-text-results --repo-type dataset --local-dir icl_compare/results_final
python writeup/figs/extract.py && python writeup/analysis.py
for f in writeup/figs/fig*.py; do PYTHONPATH=writeup/figs python "$f"; done
```

`icl_compare/run_all.py` reruns everything (data, LR sweeps, training, evaluation; about 30 A100-hours). `icl_compare/tests/test_prompt_parity.py` checks the raw-byte tasks against the paper's scripts.

## Credits

Built on [Self-Play Pretraining with Zero Data](https://arxiv.org/abs/2609.30063) ([code](https://github.com/nourya-aliz/self_play_pretraining), [weights](https://huggingface.co/nourya-cohen/solomonoff-paper)); the raw-byte tasks are a port of their harness. Text data is [DCLM-Baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0). Apache-2.0.
