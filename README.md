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
