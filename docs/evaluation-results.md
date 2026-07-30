[日本語](evaluation-results.ja.md)

# Evaluation Results

RAG quality metrics produced by the `ragas` evaluation harness (`services/eval/run_eval.py`).
Each run appends a new section below.

## How to Run

```bash
# 1. Start the stack — choose one:
make run          # includes containerised Ollama (downloads OLLAMA_MODEL on first start)
make run-remote   # without Ollama; set LLM_BACKEND=claude or openai first

# 2. Export credentials for the ragas judge (pick one; priority: Anthropic > OpenAI > Ollama)
export ANTHROPIC_API_KEY=sk-ant-…   # preferred judge: claude-haiku-4-5
export OPENAI_API_KEY=sk-…          # alternative judge (if no Anthropic key)
# No key set → falls back to Ollama running locally

# 3. Optionally configure the Query API backend
export LLM_BACKEND=claude           # or: openai / ollama

# 4. Install eval dependencies
cd services/eval && pip install -r requirements.txt

# 5. Run
make eval
# or directly: python services/eval/run_eval.py
```

## Metrics Explained

| Metric | What it measures | Needs LLM? |
|---|---|---|
| `answer_relevancy` | How relevant the generated answer is to the question | Yes |
| `faithfulness` | Whether all claims in the answer are grounded in the retrieved context | Yes |
| `context_precision` | Whether the retrieved chunks ranked highest are the most relevant | Yes |

All three metrics are computed by `ragas` using an LLM as judge (see ADR-009). The judge is selected automatically: `ANTHROPIC_API_KEY` → `claude-haiku-4-5`; `OPENAI_API_KEY` → OpenAI; neither → Ollama local fallback.
Scores range from 0 to 1; higher is better. Target thresholds: ≥ 0.70 for all three (NFR-8).

---

## Evaluation Run — 2026-04-02 13:23:39 UTC

**Samples evaluated:** 24 / 30  
**Full report:** `services/eval/results/eval_report_2026-04-02_13-23-39_UTC.json`

| Question | Ticker | Answer Relevancy | Faithfulness | Context Precision |
| --- | --- | --- | --- | --- |
| What was Apple's total net revenue for fiscal year 2024? | AAPL | 0.000 | 0.500 | — |
| What are Apple's primary supply chain risk factors disc… | AAPL | 0.819 | 0.400 | — |
| What was Apple's Services segment revenue in fiscal yea… | AAPL | 0.000 | 0.000 | — |
| How much did Apple spend on research and development in… | AAPL | 0.000 | 1.000 | — |
| Which geographic markets contribute most to Apple's net… | AAPL | 0.744 | 0.000 | — |
| What competitive risks does Apple identify in its annua… | AAPL | 0.759 | 0.900 | — |
| What was Microsoft's total revenue for fiscal year 2024? | MSFT | 0.000 | 1.000 | — |
| What was the revenue from Microsoft's Intelligent Cloud… | MSFT | 0.000 | 1.000 | 0.867 |
| What are the key competition-related risk factors Micro… | MSFT | 0.792 | 1.000 | — |
| How much did Microsoft spend on research and developmen… | MSFT | 0.000 | 0.500 | — |
| How does Microsoft describe its approach to returning c… | MSFT | 0.000 | — | — |
| What are Microsoft's three main business segments as de… | MSFT | 0.651 | 0.000 | 0.679 |
| What was Alphabet's total revenue for fiscal year 2024? | GOOGL | 0.000 | 1.000 | 1.000 |
| What was Google Cloud's revenue for fiscal year 2024? | GOOGL | 0.000 | 1.000 | 0.500 |
| What regulatory and legal risks does Alphabet disclose … | GOOGL | 0.000 | 0.250 | — |
| What is Alphabet's primary source of revenue according … | GOOGL | 0.000 | 1.000 | — |
| How much did Alphabet spend on research and development… | GOOGL | 0.000 | 1.000 | — |
| How does Alphabet describe its artificial intelligence … | GOOGL | 0.000 | 1.000 | 0.000 |
| What was Amazon's total net sales for fiscal year 2024? | AMZN | 0.000 | 1.000 | 0.583 |
| What was Amazon Web Services net sales for fiscal year … | AMZN | 0.000 | 1.000 | 0.950 |
| What competition-related risks does Amazon disclose in … | AMZN | 0.000 | 0.500 | 0.750 |
| How does Amazon describe its fulfilment and logistics n… | AMZN | 0.000 | 1.000 | 0.583 |
| What risks does Amazon identify related to its internat… | AMZN | 0.679 | 0.625 | 1.000 |
| What was Amazon's operating income for fiscal year 2024? | AMZN | 0.000 | 0.333 | 0.750 |
| **Mean** |  | **0.185** | **0.696** | **0.697** |

## Evaluation Run — 2026-07-19 12:44:53 UTC

**Samples evaluated:** 6 / 30  
**Full report:** `services/eval/results/eval_report_2026-07-19_12-44-53_UTC.json`

**Retrieval metrics (LLM-free):** section_hit_rate = 1.000 · section_mrr = 0.833 · ticker_accuracy = 1.000

| Question | Ticker | Answer Relevancy | Faithfulness | Context Precision |
| --- | --- | --- | --- | --- |
| What was Apple's total net revenue for fiscal year 2024? | AAPL | 0.966 | 1.000 | — |
| What are Apple's primary supply chain risk factors disc… | AAPL | 0.908 | 1.000 | — |
| What was Apple's Services segment revenue in fiscal yea… | AAPL | 0.000 | — | — |
| How much did Apple spend on research and development in… | AAPL | 1.000 | 1.000 | — |
| Which geographic markets contribute most to Apple's net… | AAPL | 0.989 | 1.000 | — |
| What competitive risks does Apple identify in its annua… | AAPL | 0.954 | — | — |
| **Mean** |  | **0.803** | **1.000** | **nan** |

## Evaluation Run — 2026-07-19 13:25:20 UTC

**Samples evaluated:** 30 / 30  
**Full report:** `services/eval/results/eval_report_2026-07-19_13-25-20_UTC.json`

**Retrieval metrics (LLM-free):** section_hit_rate = 0.800 · section_mrr = 0.678 · ticker_accuracy = 1.000

| Question | Ticker | Answer Relevancy | Faithfulness | Context Precision |
| --- | --- | --- | --- | --- |
| What was Apple's total net revenue for fiscal year 2024? | AAPL | 1.000 | 1.000 | 0.667 |
| What are Apple's primary supply chain risk factors disc… | AAPL | 0.913 | 1.000 | 0.000 |
| What was Apple's Services segment revenue in fiscal yea… | AAPL | 0.000 | 0.750 | 0.000 |
| How much did Apple spend on research and development in… | AAPL | 1.000 | 1.000 | 0.833 |
| Which geographic markets contribute most to Apple's net… | AAPL | 0.999 | 1.000 | 0.500 |
| What competitive risks does Apple identify in its annua… | AAPL | 0.936 | 1.000 | 1.000 |
| What was Microsoft's total revenue for fiscal year 2024? | MSFT | 1.000 | 1.000 | 0.667 |
| What was the revenue from Microsoft's Intelligent Cloud… | MSFT | 1.000 | 1.000 | 0.250 |
| What are the key competition-related risk factors Micro… | MSFT | 0.924 | 0.947 | 0.000 |
| How much did Microsoft spend on research and developmen… | MSFT | 1.000 | 1.000 | 0.917 |
| How does Microsoft describe its approach to returning c… | MSFT | 0.946 | 1.000 | 0.500 |
| What are Microsoft's three main business segments as de… | MSFT | 0.774 | 1.000 | 1.000 |
| What was Alphabet's total revenue for fiscal year 2024? | GOOGL | 0.999 | 1.000 | 0.700 |
| What was Google Cloud's revenue for fiscal year 2024? | GOOGL | 1.000 | 1.000 | 0.333 |
| What regulatory and legal risks does Alphabet disclose … | GOOGL | 0.969 | 1.000 | 0.500 |
| What is Alphabet's primary source of revenue according … | GOOGL | 0.902 | — | 0.000 |
| How much did Alphabet spend on research and development… | GOOGL | 1.000 | 1.000 | 1.000 |
| How does Alphabet describe its artificial intelligence … | GOOGL | 0.823 | 0.700 | 0.000 |
| What was Amazon's total net sales for fiscal year 2024? | AMZN | 0.958 | 1.000 | 1.000 |
| What was Amazon Web Services net sales for fiscal year … | AMZN | 0.000 | 1.000 | 0.000 |
| What competition-related risks does Amazon disclose in … | AMZN | 0.967 | 0.893 | 0.000 |
| How does Amazon describe its fulfilment and logistics n… | AMZN | 0.876 | 0.583 | 0.000 |
| What risks does Amazon identify related to its internat… | AMZN | 0.978 | 1.000 | 0.500 |
| What was Amazon's operating income for fiscal year 2024? | AMZN | 1.000 | 1.000 | 1.000 |
| What was JPMorgan Chase's total net revenue for fiscal … | JPM | 1.000 | 1.000 | 1.000 |
| What was JPMorgan Chase's net interest income for fisca… | JPM | 0.000 | 1.000 | 0.000 |
| What credit risk factors does JPMorgan Chase identify i… | JPM | 0.925 | 1.000 | 0.000 |
| How does JPMorgan Chase describe its capital management… | JPM | 0.000 | 1.000 | 0.000 |
| What was JPMorgan Chase's provision for credit losses i… | JPM | 1.000 | 1.000 | 0.000 |
| What are JPMorgan Chase's four main business segments a… | JPM | 0.859 | 0.625 | 0.000 |
| **Mean** |  | **0.825** | **0.948** | **0.412** |

## Evaluation Run — 2026-07-25 21:00:40 UTC

**Samples evaluated:** 30 / 30  
**Full report:** `services/eval/results/eval_report_2026-07-25_21-00-40_UTC.json`

**Retrieval metrics (LLM-free):** section_hit_rate = 0.867 · section_mrr = 0.759 · ticker_accuracy = 1.000

| Question | Ticker | Answer Relevancy | Faithfulness | Context Precision |
| --- | --- | --- | --- | --- |
| What was Apple's total net revenue for fiscal year 2024? | AAPL | 0.999 | 1.000 | 0.833 |
| What are Apple's primary supply chain risk factors disc… | AAPL | 0.922 | 1.000 | 0.333 |
| What was Apple's Services segment revenue in fiscal yea… | AAPL | 1.000 | 1.000 | 0.367 |
| How much did Apple spend on research and development in… | AAPL | 1.000 | 1.000 | 1.000 |
| Which geographic markets contribute most to Apple's net… | AAPL | 0.981 | 0.778 | 0.500 |
| What competitive risks does Apple identify in its annua… | AAPL | 0.936 | 1.000 | 0.333 |
| What was Microsoft's total revenue for fiscal year 2024? | MSFT | 1.000 | 1.000 | 0.887 |
| What was the revenue from Microsoft's Intelligent Cloud… | MSFT | 0.976 | 1.000 | 0.500 |
| What are the key competition-related risk factors Micro… | MSFT | 0.902 | 0.941 | 0.000 |
| How much did Microsoft spend on research and developmen… | MSFT | 1.000 | 0.800 | 1.000 |
| How does Microsoft describe its approach to returning c… | MSFT | 1.000 | 0.778 | 1.000 |
| What are Microsoft's three main business segments as de… | MSFT | 0.885 | 0.857 | 0.887 |
| What was Alphabet's total revenue for fiscal year 2024? | GOOGL | 1.000 | 1.000 | 0.950 |
| What was Google Cloud's revenue for fiscal year 2024? | GOOGL | 0.959 | 0.500 | 0.200 |
| What regulatory and legal risks does Alphabet disclose … | GOOGL | 0.956 | 1.000 | 0.000 |
| What is Alphabet's primary source of revenue according … | GOOGL | 0.865 | 1.000 | 0.167 |
| How much did Alphabet spend on research and development… | GOOGL | 1.000 | 1.000 | 1.000 |
| How does Alphabet describe its artificial intelligence … | GOOGL | 0.907 | 1.000 | 0.000 |
| What was Amazon's total net sales for fiscal year 2024? | AMZN | 0.996 | 1.000 | 1.000 |
| What was Amazon Web Services net sales for fiscal year … | AMZN | 0.939 | 1.000 | 0.500 |
| What competition-related risks does Amazon disclose in … | AMZN | 0.849 | 1.000 | 0.000 |
| How does Amazon describe its fulfilment and logistics n… | AMZN | 0.852 | 1.000 | 0.000 |
| What risks does Amazon identify related to its internat… | AMZN | 0.979 | 1.000 | 1.000 |
| What was Amazon's operating income for fiscal year 2024? | AMZN | 0.999 | 1.000 | 0.833 |
| What was JPMorgan Chase's total net revenue for fiscal … | JPM | 0.964 | 0.769 | 1.000 |
| What was JPMorgan Chase's net interest income for fisca… | JPM | 0.000 | 0.857 | 0.000 |
| What credit risk factors does JPMorgan Chase identify i… | JPM | 0.889 | 1.000 | 0.000 |
| How does JPMorgan Chase describe its capital management… | JPM | 0.852 | 1.000 | 0.000 |
| What was JPMorgan Chase's provision for credit losses i… | JPM | 1.000 | 1.000 | 0.200 |
| What are JPMorgan Chase's four main business segments a… | JPM | 0.903 | 1.000 | 0.000 |
| **Mean** |  | **0.917** | **0.943** | **0.483** |
