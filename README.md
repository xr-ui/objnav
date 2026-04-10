# CRG-Nav: Confidence-Aware Reasoning with Geometric Verification for Zero-Shot Object Navigation
---
 
## Overview
 
CRG-Nav addresses three core limitations of existing zero-shot object navigation methods:
 
- **Unreliable long-range perception** — distant observations are down-weighted via a Gaussian-decay confidence mechanism
- **Hallucination-induced decision bias** — perception (VLM + YOLO-World) is decoupled from decision-making (LLM), preventing perception errors from polluting high-level reasoning
- **Inaccurate stopping decisions** — a dual semantic-geometric verification module uses depth data to estimate 3D target distance before triggering a stop
<img width="800"  alt="0e87e393-7fca-49ed-9d55-d5be3f789156" src="https://github.com/user-attachments/assets/3f0031e7-f3d7-4568-8f7c-b180b4729f84" />

## Getting start
