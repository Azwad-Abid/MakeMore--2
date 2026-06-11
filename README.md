# Makemore Part 2: Multi-Layer Perceptron

## Results
- Final NLL: 2.0949
- Improvement over bigram baseline: 14%
- Generated names: anuelen, tia, marian, davius, saila, yana, kemah, lavin, emiah, aden

## Architecture
- Context window: 3 characters
- Embedding dimension: 10
- Hidden layer: 200 neurons
- Training: 10 epochs, batch size 32

## Key learnings
- Depth matters (hidden layer improved results significantly)
- Context window helps predictions
- Sampling generates realistic new names
