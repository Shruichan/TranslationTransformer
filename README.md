# TranslationTransformer

English → Japanese and English → French translation models, built on top of a multilingual BERT
encoder-decoder. This started as a "can I fine-tune BERT to translate?" experiment and ended up
with two working models, a pile of training logs, and a couple of throwaway plotting scripts that
somehow survived.

## What's in here

```
src/
  train_japanese.py        # trains the EN→JA model
  train_french.py          # trains the EN→FR model
  translate_japanese.py    # loads the saved JA model and translates a sentence
  translate_french.py      # same, for FR
  plot_training.py         # quick batch-loss / accuracy plot from a log file
data/
  jpn.txt                  # Tatoeba EN/JA pairs
  fra.txt                  # Tatoeba EN/FR pairs
logs/
  training_results_*.txt   # per-batch loss + accuracy for both runs
  graph_*.txt              # filtered logs used by plot_training.py
```

The trained `.pth` files aren't in the repo (they're big and the `.gitignore` keeps them out).
Train your own with the scripts, or wire it up to one you already have.

## The model

It's a `transformers.EncoderDecoderModel` with `bert-base-multilingual-cased` on both sides. The
same tokenizer handles English on the way in and Japanese/French on the way out, which is the
whole reason for using the multilingual checkpoint instead of plain BERT.

Training settings are the same for both languages:

- batch size 16
- 3 epochs
- AdamW, lr=5e-5
- 90/10 train/val split, max sequence length 128

Pairs come from Tatoeba (`tatoeba.org`). The cleaning step in both training scripts trims
everything after the first sentence terminator (`. ? ! 。 ？ ！`) so the model isn't trying to
emit multi-sentence outputs.
