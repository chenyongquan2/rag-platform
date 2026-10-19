"""Rerank-threshold setting from recorded rerank scores (spec P2-02; design s10-2, s14-4).

`collect` stores every reranked candidate's score per question (threshold 0); `sweep` reads those
files and reports, for each candidate threshold, the false-refusal rate on answerable questions and
the wrong-answer-accepted rate on no-answer questions with Wilson intervals.
"""
