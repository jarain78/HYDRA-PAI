
# The Hybrid Mind: Integrating Cosmos Reason2, Ontological Memory, and Symbolic Planning for Explainable Physical AI


<p align="center">
  <img src="static/HYDRA_PAI_Logo.png" alt="HYDRA-PAI Logo" width="520"/>
</p>

**Physical AI needs more than perception.**  
This project demonstrates a hybrid AI architecture where:

- **Cosmos Reason2** performs high-level visual reasoning.
- **OWL ontologies** provide structured, hierarchical knowledge representation.
- **CLIPS (clipspy)** performs deterministic symbolic planning with an inference engine.
- **PyBullet** executes the plan via robotic manipulation (IK + smooth motion + grasp constraints).

**Result:** The robot does not merely describe what it sees — it understands what should happen, plans symbolically, and executes physical actions accordingly.

---

## 🧠 Why This Matters (NVIDIA Cosmos Cookoff Focus)

Most Vision-Language robotics demos stop at captioning or labeling.  
This project bridges the critical gap:

> **Perception → Reasoning → Knowledge → Planning → Physical Action**

- Cosmos Reason2 provides powerful foundation-level visual reasoning.
- Ontologies enforce structured, explainable domain knowledge (rooms, objects, hierarchy).
- CLIPS ensures auditable, rule-based decision-making.
- The robot executes tasks based on explainable logic — not black-box outputs.

This architecture demonstrates how **foundation models become physically actionable when grounded in structured knowledge and symbolic reasoning.**

---

## 🔬 System Architecture

## 🧩 HYDRA-PAI Block Diagram

![HYDRA-PAI Flowchart](static/HIDRA_PAI_Flowchart.png)


```
PyBullet Camera (RGB)
        |
        v
Perception (YOLO) → Grounding (detection → body_id)
        |
        v
Ontology (OWLReady2)
  - Object properties: locatedIn, canBePickedBy
  - Class hierarchy: subClassOf
        |
        v
Cosmos Reason2 (High-Level Reasoner)
  - Input: image + ontology summary + perception facts
  - Output: JSON intents (move/ignore, priority, rationale)
        |
        v
CLIPS (clipspy) Planner
  - Input: (obj ...) + (intent ...)
  - Output: ordered (task ...) facts + traces
        |
        v
PyBullet Executor (IK-based manipulation)
```

---

## 🚀 Demo Flow

1. PyBullet renders a robotic scene (Panda or MyCobot + YCB objects).
2. YOLO detects objects and associates them with simulated bodies.
3. The ontology provides:
   - destination room (`locatedIn`)
   - pickability (`canBePickedBy`)
   - class hierarchy (subClassOf relationships)
4. Cosmos Reason2 receives:
   - a camera frame
   - ontology summary
   - grounded object state
   and produces structured JSON intents.
5. CLIPS transforms intents into executable symbolic tasks.
6. The robot performs smooth pick-and-place operations.

---

## ✅ Key Contributions

### 1️⃣ Cosmos as a True High-Level Reasoner
Cosmos is constrained to produce structured JSON:

```
{
  "high_level_goal": "...",
  "intents": [
    {"onto": "cup", "action": "move", "to_room": "kitchen", "priority": 1, "rationale": "..."}
  ]
}
```

This prevents free-form responses and ensures reliable integration.

---

### 2️⃣ Ontology-Grounded Decision Making

Cosmos reasons over:

- Object → class relationships
- Class hierarchy chains
- Ontology constraints (destination, pickability)
- Current room vs desired room

Expanding the ontology automatically expands the robot’s knowledge without retraining models.

---

### 3️⃣ Deterministic Symbolic Planning (CLIPS)

CLIPS converts high-level intents into tasks using inference rules:

- Not pickable → ignore
- Missing destination → ignore
- Out-of-place → move
- Already correct → ignore

All reasoning steps are traceable and auditable.

---

### 4️⃣ Real Physical Execution

The system closes the loop:

> See → Understand → Decide → Plan → Act

---

## 📂 Repository Contents

- `ontoai_gui_ycb_cosmos_clips.py` – Streamlit GUI integrating Cosmos + Ontology + CLIPS + PyBullet
- `hogar_en.owl` – Example ontology (rooms, objects, hierarchy)
- YCB objects via PyBullet simulation

---

## ⚙️ Quickstart

### Install Dependencies

```bash
pip install streamlit pybullet owlready2 clipspy ultralytics transformers torch imageio imageio-ffmpeg pillow
```


---

## 🔑 Hugging Face Access Requirement

**Cosmos Reason2 models are hosted on Hugging Face and require authentication to download.**

Before running the project, you must create a **Hugging Face access token**.

### 1. Create an Access Token

Create or log in to your Hugging Face account:

https://huggingface.co

Then generate a token:

https://huggingface.co/settings/tokens

Create a token with **Read access**.

### 2. Login from the Terminal

Install the Hugging Face CLI if needed:

```bash
pip install huggingface_hub
```

Authenticate with:

```bash
huggingface-cli login
```

Paste your **access token** when prompted.

### 3. Alternative: Environment Variable

You can also export the token manually:

```bash
export HF_TOKEN=your_huggingface_token_here
```

### 4. Why This Is Required

The **Cosmos Reason2 model weights are gated**, so authentication is required before the Transformers library can download them.

Once authenticated, the model will be automatically downloaded and cached locally by Hugging Face.


### Run the GUI

```bash
streamlit run src/ontoai_dashboard_streamlit_mycobot280.py
```

### In the GUI

- Enable **Cosmos High-Level Reasoning**
- Load ontology (`.owl`)
- Enable execution
- Observe planning traces and robot motion

---

## 🧩 Why Ontologies?

OWL ontologies provide:

- Hierarchical reasoning
- Domain extensibility
- Explainable constraints
- Cross-domain adaptability

To expand capabilities, edit the ontology — not the neural model.

---

## 🧪 Evaluation for the Cookoff

### Baseline (Vision-only)
- Cosmos captioning
- No structured task execution

### Hybrid Mind (Full Stack)
- Cosmos + Ontology + CLIPS
- Explicit intents
- Planned symbolic tasks
- Physical execution

Compare:

- Explainability
- Determinism
- Robustness
- Extensibility

---

## 🔮 Future Work

- Multi-object scheduling optimization
- Safety constraints from ontology (fragile, hazardous, heavy)
- Multi-robot reasoning
- Real robot integration (ROS2)
- Persistent knowledge updates (ontology learning)

---

## 📌 Submission Summary

This project presents a hybrid Physical AI architecture integrating NVIDIA Cosmos Reason2 with OWL ontologies and CLIPS symbolic planning. Cosmos acts as a high-level visual reasoner grounded on structured knowledge extracted from an ontology that encodes object-room relationships, pickability, and class hierarchy. The model outputs structured JSON intents, which are transformed into symbolic facts and processed by a CLIPS inference engine to generate ordered task plans with full traceability. A robotic manipulator in PyBullet executes these plans using inverse kinematics and smooth motion control. This integration demonstrates how foundation models gain robustness, explainability, and extensibility when paired with explicit knowledge representation and symbolic reasoning. Rather than relying solely on black-box perception, the system closes the loop from perception to physical action through interpretable planning. The Hybrid Mind showcases a scalable blueprint for trustworthy household and assistive robotics.

---

## 📜 License

Recommended:
- **CC BY-NC-SA 4.0** (non-commercial sharing)
or
- **AGPL-3.0** (strong copyleft for code)

---

## 🙏 Acknowledgements

- NVIDIA Cosmos Reason2
- OWLReady2
- CLIPS / clipspy
- PyBullet
- Ultralytics YOLO