"""How often the sufficiency judge agrees with a label made from the question sets (spec P3-03;
design s13-4, s14-2, s14-4).

The label needs no human: a question is "enough on round one" when its labelled gold page is
already among the chunks round one returned (reranked, cut by the rerank threshold), and a
no-answer question is never enough. `run` retrieves every question through the real search chain,
asks the judge (`rag.agent.judge.judge_rounds`) and records verdict and label; `report` computes
accuracy, precision and recall of "enough" with Wilson intervals, the confusion matrix, the split
by question type and the comparison with the rules alone and with always running two rounds.
"""
