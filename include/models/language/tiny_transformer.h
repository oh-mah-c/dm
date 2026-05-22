#ifndef DM_TINY_TRANSFORMER_H
#define DM_TINY_TRANSFORMER_H

/*
 * TinyStories-style tiny byte language model.
 *
 * This is the first small-model module for dm: a compact C99 causal LM that
 * trains directly on bytes, reports loss/perplexity, saves checkpoints, and
 * generates text. It is intentionally small enough for research smoke tests
 * and data-pipeline validation before introducing heavier GPT-Neo/Mamba/ViT
 * style implementations.
 */

int dm_tiny_transformer_cli(int argc, char **argv);

#endif /* DM_TINY_TRANSFORMER_H */
