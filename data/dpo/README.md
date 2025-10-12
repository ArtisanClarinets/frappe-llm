# Frappe DPO Dataset Guidelines

This directory hosts paired-preference datasets used for Direct Preference Optimization (DPO) and
other pairwise alignment objectives. The pipeline supports two complementary JSONL schemas:

1. **Chat format** – each record contains a conversation `messages` array (OpenAI style) with
   assistant completions denoted as `chosen` and `rejected`. This mirrors Axolotl's
   `chat_template` workflows and is ideal when preference annotation happens inside the product.
2. **Simple prompt format** – a flat object with `prompt`, `chosen`, and `rejected` strings. The
   Axolotl configuration in this repo defaults to this schema for lightweight curation or when
   exporting comparisons from notebooks.

Export preference pairs from your review tooling into `frappe_pairs.jsonl` using one of the formats
above. Keep the following governance practices in mind:

- Never include customer PII, API keys, or credentials in prompts or completions.
- Exclude SQL or shell snippets that could be misused outside Frappe's sandbox.
- Ensure rejected answers genuinely demonstrate why the chosen answer is superior (missing steps,
  hallucinations, or unsafe migrations are common counterexamples).

The repository ships with `frappe_pairs.sample.jsonl` to validate the training stack. Replace it with
real annotations before launching DPO runs.
