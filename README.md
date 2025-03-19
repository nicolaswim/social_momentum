# 🚀 Social Momentum Navigation

This project implements the **Social Momentum (SM) algorithm**, inspired by the work of **Mavrogiannis et al. (2022)**, titled:

> **"Social Momentum: Design and Evaluation of a Framework for Socially Competent Robot Navigation"**  
> 📄 *(See `Mavrogiannis et al. - 2022 - Social Momentum Design and Evaluation of a Framework for Socially Competent Robot Navigation.pdf` in this directory.)*

The **goal** of this project is to develop a **Python-based simulation** of the Social Momentum algorithm, which enables **legible robot motion in human environments** by optimizing **angular momentum-based collision avoidance**.

---
## 📂 **Project Structure**

---

## ⚙️ **Installation & Setup**

### **1️⃣ Install Dependencies**
This project requires **Python 3.8+**. First, activate the virtual environment:

```bash
# Navigate to the project directory
cd ~/Documents/social_momentum

# Run the setup script
bash setup.bash
```

This will:
	•	Create the required folders and files.\
	•	Set up a Python virtual environment (venv/).\
	•	Install necessary Python dependencies (numpy, matplotlib).\

2️⃣ Activate the Virtual Environment

If not already active:
```python
source venv/bin/activate
```

🚀 Running the Simulation

Once installed, run:

```python
python src/main.py
```

This will simulate a robot moving in a 2D space, avoiding human agents using Social Momentum-based collision avoidance.

Sample output:

```
🚀 Starting Social Momentum Simulation
Step 1: Robot Position = [0.2 0.1]
Step 2: Robot Position = [0.4 0.3]
...
```

📖 Reference Paper

This project is based on the paper:
	•	Mavrogiannis, C., Alves-Oliveira, P., Thomason, W., & Knepper, R. (2022)
“Social Momentum: Design and Evaluation of a Framework for Socially Competent Robot Navigation.”
📄 (Available in this directory: Mavrogiannis et al. - 2022 - Social Momentum Design and Evaluation of a Framework for Socially Competent Robot Navigation.pdf)

For more details, refer to the official ACM paper:
🔗 https://doi.org/10.1145/3495244