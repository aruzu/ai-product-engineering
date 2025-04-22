# AI Product Research Interview Simulator

An interactive CLI tool that **simulates a multi–persona product‑market interview** powered by OpenAI models.  
It lets you quickly stress‑test a product idea, capture rich qualitative feedback and walk away with an *executive summary* and *sentiment analysis* – all in a single run.

---

## Features

* **Facilitated focus‑group** – an AI *facilitator* asks core and follow‑up questions.
* **Multiple personas** – configurable agents (e.g. *Alice – Athlete*, *Bob – Office Worker*) respond, react to each other and reference personal context.
* **Rich console output** – colour‑coded panels and tables powered by [`rich`](https://github.com/Textualize/rich).
* **Automatic analysis**  
  * Sentiment analysis for every persona.  
  * One‑click *GO / NO‑GO* decision and key rationale.

---

## Quick‑start

1. **Clone** the repository and `cd` into it:

   ```bash
   git clone <repo-url>
   cd <repo>
   ```

2. **Install** the dependencies (Python ≥ 3.9):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set your OpenAI key** (required by the `openai-agents` runtime):

   ```bash
   export OPENAI_API_KEY=sk-...        # PowerShell: setx OPENAI_API_KEY "sk-..."
   ```

4. **Run** the interview:

   ```bash
   python userboard4-baimuratov.py
   ```

You will see colour‑coded dialogue followed by an executive summary report.

---

## Customising the session

Open `userboard4-baimuratov.py` and edit the variables at the bottom of the file:

```python
topic = "A subscription‑based smart water bottle that reminds users to drink."

personas = [
    {"name": "Alice (Athlete)", "description": "26‑year‑old marathon runner; tracks hydration closely"},
    ...
]

core_qs = [
    "What is your initial reaction to the idea?",
    ...
]
```

* `topic` – the product / feature you want to test.
* `personas` – as many personas as you like (name + short description).
* `core_qs` – ordered list of questions the facilitator must ask.
* `max_followups` – number of dynamic follow‑up questions (default **3**).

---

## File structure

```
├── userboard4-baimuratov.py   # Main entry‑point
├── requirements.txt           # Python dependencies
└── README.md                  # You are here
```

---

## Contributing

Pull requests are welcome! Feel free to open issues for bugs or feature ideas.

---

## License

Released under the MIT License – see [`LICENSE`](LICENSE) for details.
