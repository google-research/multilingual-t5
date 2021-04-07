These .gin files define input and target lengths for all tasks.

1. Input length: set to a number that is a power of two and
   larger than 1.1 * 99.9 percentile sequence length in train, dev, and test.

2. Target length: computed by packing examples from train, valid, and test,
   and taking the max packed target length.

