# NOTICE: huggingface_doc_qa_eval.jsonl

`huggingface_doc_qa_eval.jsonl` is derived from the Hugging Face dataset
[`m-ric/huggingface_doc_qa_eval`](https://huggingface.co/datasets/m-ric/huggingface_doc_qa_eval),
revision `5f70aa9a1e2430f528ac3f27f01f0ba8719c0704` (file `data/train-00000-of-00001.parquet`,
pinned with its size and sha256 in `data/manifests/hf_docs.json`).

The source dataset is licensed under the Apache License, Version 2.0:
<https://www.apache.org/licenses/LICENSE-2.0>. This derived file is distributed under the same
license.

## Changes made to the source

Written by `uv run python -m rag.eval.retrieval export-hf` (spec P0-10):

- Kept the columns `question`, `answer` and `source_doc`; every other column is dropped,
  including `context` (the passage each question was generated from) and the three filter
  scores with their explanations.
- `question` and `answer`: leading and trailing whitespace stripped (most questions end with a
  newline). The text is otherwise unchanged.
- `source_doc` became the page label `relevant_pages: ["hf_docs:<source_doc>"]` (the corpus doc_id
  scheme of spec P0-03); `relevant_chunks` is empty.
- Added `qid` = `huggingface_doc_qa_eval:<row index>` (row order of the pinned file).
- Format changed from parquet to JSON Lines, one question per line.
