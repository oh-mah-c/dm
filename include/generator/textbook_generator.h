#ifndef DM_TEXTBOOK_GENERATOR_H
#define DM_TEXTBOOK_GENERATOR_H

/*
 * Phi/Textbooks-Are-All-You-Need style corpus builder.
 *
 * Paper: "Textbooks Are All You Need II: phi-1.5 technical report",
 * arXiv:2309.05463v1.
 *
 * Implements the reproducible local parts of the data pipeline:
 *   - topic-seeded synthetic textbook prompt generation
 *   - optional web-sample snippets for diversity
 *   - exercise/answer prompt variants
 *   - textbook-quality heuristic scoring/filtering of generated text
 *   - corpus mixing with phi-1.5-style synthetic/web/code ratios
 *
 * The paper uses existing LLMs to synthesize the actual textbooks. This
 * module emits prompts and filters/mixes returned generations; it does not
 * fabricate those LLM generations locally.
 */

int dm_textbook_generator_cli(int argc, char **argv);

#endif /* DM_TEXTBOOK_GENERATOR_H */
