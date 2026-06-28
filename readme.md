# AlaynTechnologies: Natively Intelligent Optical Automation

**AlaynTechnologies** is the first fully integrated, natively smart optical hardware ecosystem. By co-designing proprietary smart kinematic mounts with a Software-Defined MIMO orchestration layer, we turn weeks of manual laser alignment into 45 seconds of autonomous execution.



## 🚀 The Problem: The Cascaded Alignment Bottleneck
Aligning a multi-mirror optical train is a non-linear physics problem—adjusting one mirror ruins the alignment of the others (cross-talk).
* **The Manual Cost:** Labs waste weeks of expensive PhD labor manually walking the beam.
* **The Legacy Hardware Trap:** Existing automation solutions (MIMO Hexapods) are closed-ecosystem, proprietary, and cost upwards of $100,000 per setup. 

## 💡 The Full-Stack Solution
OptiAlign removes the bottleneck by providing both the hardware and the brain:
1. **Smart Kinematic Mounts:** Our custom motorized holders use a **Reaction Wheel (Momentum Wheel) mechanism**. By spinning an internal rotor, we exert torque directly on the micrometer knob. This eliminates "axial binding" and removes the need for physical bracing against the optical table.
2. **Software-Defined MIMO Controller:** Our optimization agent uses **Nelder-Mead Optimization** to calculate the simultaneous adjustments needed across the entire optical train, treating the setup as a coordinated system rather than a series of independent knobs.

## ⚙️ How It Works (The Physics)
The system leverages the **Conservation of Angular Momentum**:
* When our internal motor spins the rotor (the propeller) in one direction, an equal and opposite reaction torque is applied to the housing. 
* Because our device is isolated on the micrometer screw, this torque precisely rotates the knob in the opposite direction. 
* **Result:** High-precision, vibration-free actuation that is fully plug-and-play.

## 📈 Business Model: Hardware-Enabled SaaS
* **The Hook:** Deliver a system for free to 10 laboratories
* **The Margin:** After succussfull lauch, we convert to subscription model and annual recurring revenue system (15k ARR)
* **The Growth:** Backed by data, we advertise and spread our system into labs all over the world (20-100M+ ARR)

## 🛠️ MVP
We have implemented a rigorous **Digital Twin** to prove our math:
* **The Brain (software optimization):** Uses `LightPipes` visualize laser and mirrors, and uses Nelder-Mead optimization for finding the optimal positions of the mirrors
* **Verification:** The simulation proves our algorithms effectively control a cascaded system before we even mount the hardware.
* **3D model and visualization:** Blender script that creates laser, adjustable mirror and hardware device that will turn the knob on the mirroe. Rendered simulation to show how it works.
* **Verification:** Science backed technology, with technical plan / roadmap built to support our solution.

## ⚙️ Installation & Usage - Software simulation

### Prerequisites
* Python 3.10+
* Git
* C++ Build Tools (required for `LightPipes` compilation)

### Setup
1. Clone the repository:
    git clone [https://github.com/aleksamarusic/AlaynTechnologies.git](https://github.com/aleksamarusic/AlaynTechnologies.git)
    cd AlaynTechnologies

2. Create and activate a virtual environment:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
    pip install -r requirements.txt

4. Run the software simulation:
    python alignment_software_simulation.py


## ⚙️ Installation & Usage - Hardware simulation

### Prerequisites
* Blender 5.1
* Git

### Setup
1. Clone the repository:
    git clone [https://github.com/aleksamarusic/AlaynTechnologies.git](https://github.com/aleksamarusic/AlaynTechnologies.git)
    cd AlaynTechnologies

2. Install dependencies:
    [Download and install Blender](https://www.blender.org/download/)

4. Run Blender, go to scripting, open blender_animation_script.py and run

---
*Built for the Paris Builds Hackathon (June 2026).*