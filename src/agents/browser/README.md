# Agentic Browsing System Overview

This system integrates a multi-agent architecture designed to autonomously plan, execute, and refine web-based research tasks. It combines a research-oriented language model (SLAI LM) with a Planning Agent and a Browser Agent, all coordinated to serve complex user queries efficiently.

---

## System Components

### 1. User Interface

* The user provides **queries or tasks**.
* Optionally, the user may provide **feedback** (skipped in the first iteration).

### 2. Research Model + SLAI LM

* Processes the incoming query.
* Interfaces with:

  * **Language Agent**: Handles linguistic analysis.
  * **Text Encoder**: Converts text into embeddings.
  * **Tokenizer**: Breaks text into subword units.
* Passes the structured goal to the Planning Agent.

---

## Planning Agent Flow

1. **Receive Goal Task**
2. **Check Task Type**

   * If primitive → check preconditions, execute, apply effects, determine success/fail.
   * If abstract → retrieve matching methods.
3. **Score Methods**

   * Uses decision tree (DT) and gradient boosting (GB) heuristics.
4. **Select Best Method**

   * Chooses the method with the highest heuristic score.
5. **Decompose into Subtasks**
6. **Recursively Decompose** (if needed)
7. **Simulate Effects** on the world state.
8. **Build Full Executable Plan**
9. **Schedule Tasks** using the DeadlineAwareScheduler.
10. **Execute Plan** → Monitor for success or failure.
11. **Failure Handling**

    * If failure: apply Bayesian + grid search to find alternative methods → replan.
12. **Periodic Retraining**

    * Every N plans, retrains on accumulated data.

---

## Browser Agent Flow

1. **Receive User Query or Plan**
2. **Initialize Browser** (typically headless Chrome)
3. **NLP Parse on Query**
4. **Google Search Navigation**
5. **CAPTCHA Handling** (if detected)
6. **Select Best-Matching Link** (based on relevance)
7. **Click and Load Content**
8. **Capture Page Snapshot** (HTML, text, screenshot)
9. **Run Content Handlers**

   * For PDFs, arXiv papers, or other special formats.
10. **Apply Reasoning Agent**

    * Extracts structured facts.
11. **Send Data to Learning Agent**

    * Records relevance, updates model weights.
12. **Return Analyzed Results** to the user.
13. **Failure Handling**

    * If CAPTCHA blocks or hard failures occur: fallback or retry with exponential backoff (up to 5 recursive retries).

---

## Skills (Tools) Available to Browser Agent

* `scroll()`
* `click()`
* `navigate()`
* `entertext()`
* `get_dom()`
* `get_url()`
* `google_search()`
* `open_url()`
* `pdf_extractor()`
* `press_key()`

These tools let the Browser Agent dynamically interact with web pages, extract needed content, and adapt to complex tasks.

---

## System Highlights

* Modular, multi-agent design.
* Combines heuristic planning with execution and real-time feedback.
* Can recover from failures, handle dynamic web environments (e.g., CAPTCHAs).
* Integrates learning agents for continuous improvement.
* Can be expanded with more tools, models, or domains.

