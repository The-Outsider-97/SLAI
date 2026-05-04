# Project: MindWeave

**A Web-Based Cognitive and Socio-Emotional RPG Simulation**

Project MindWeave is an interactive, browser-based digital environment engineered at the intersection of cognitive neuroscience and interactive game design. It is featured as a premier module within the R-Games suite. Unlike traditional cognitive applications that suffer from the near-transfer problem, MindWeave targets fluid intelligence (IQ), executive function, and emotional quotient (EQ) through far-transfer neuroplasticity, progressive overload, and rich socio-emotional contexts.

## Table of Contents

- [Overview](#overview)
- [Scientific Foundation](#scientific-foundation)
- [Architecture & Technologies](#architecture--technologies)
- [Project Structure](#project-structure)
- [Configuration (LLM API Setup)](#configuration-llm-api-setup)
- [Usage](#usage)
- [License](#license)

## Overview

Players assume the role of a "Weaver," a diplomat-engineer tasked with rebuilding a fractured sci-fi society. The platform wraps scientifically validated cognitive tasks within a narrative, engaging the brain's dopamine reward pathways to enhance learning and memory retention. 

## Scientific Foundation

*   **Campaign Progression:** Structured as Mission Briefing → IQ Systems Repair → EQ Diplomacy & Regulation → Metacognitive Debrief, with additional protocol drills for task-switching, planning optimization, cognitive reappraisal, and co-regulation boundaries.
*   **Evidence-Informed Design:** Protocols are aligned with research-backed principles from working-memory training, executive control, socio-emotional learning, cognitive reappraisal, and reflective transfer practice.
*   **IQ & Problem-Solving Engine:** Utilizes advanced implementations of the Dual N-Back task, resource management planning, and procedural logic gates to target working memory, cognitive flexibility, and pattern recognition.
*   **EQ Engine:** Employs the Facial Action Coding System (FACS) to train micro-expression recognition. Features empathy bridging via active listening and biometric emotional regulation.
*   **Metacognitive Integration:** Features a debriefing phase requiring players to articulate their strategies, bridging the gap between in-game mechanics and real-world application.

## Architecture & Technologies

MindWeave operates within the R-Games ecosystem. It relies on a centralized Python server for game routing, while the MindWeave module handles its own specific AI logic and frontend rendering.

*   **Frontend Framework:** Vanilla JS / HTML5 (served via the R-Games structure)
*   **3D Rendering:** Three.js (for isometric world and FACS facial animations)
*   **Backend Server:** Python (FastAPI / Flask) via `app.py`
*   **Module Logic:** `ai_mindweave.py` handles backend validation and LLM integration.
*   **API Key Management:** Client-side local storage (Zero-knowledge backend storage).

## Project Structure

```text
/R-Games
│
├─ app.py                 # Central server (localhost:8000)
├─ index.html             # R-Games Launcher
├─ ai_mindweave.py        # MindWeave backend logic
│
└─ mindweave/
   ├─ index.html          # MindWeave frontend client
   └─ ...                 # Three.js assets and scripts
```

## Configuration (LLM API Setup)

To facilitate dynamic, empathetic conversations with NPCs, MindWeave requires a Large Language Model API. 

This project utilizes a strict "Bring Your Own Key" (BYOK) architecture to ensure user privacy. The backend (`app.py` / `ai_mindweave.py`) does not require or store your API key. 
When launching Project: MindWeave from the R-Games launcher, the frontend will prompt the user to input their LLM API Key via a secure modal. This key is stored exclusively in the browser's `localStorage` and is passed directly to the API provider or securely in request headers to `ai_mindweave.py` during runtime.

## Usage

1. Start the central R-Games server from the root directory:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:8000`.

3. Select **Project: MindWeave** from the launcher grid. If this is your first time, you will be prompted to enter your LLM API Key.

*Note for 3D Rendering Performance:* Ensure hardware acceleration is enabled in your web browser to properly render the Three.js scenes and micro-expressions.

## License

This project is licensed under the MIT License.
```
