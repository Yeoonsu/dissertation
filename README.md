### Dissertation Repository Description

**Repository Name:** *Optimized Data Preparation for Predictive Process Monitoring Using Large Language Models (LLMs)*

---

#### **Overview**
This repository contains the full implementation, data, and results supporting the dissertation:  
**"Optimized Data Preparation Pipelines for Predictive Process Monitoring Using LLMs."**

Predictive Process Monitoring (PPM) relies heavily on high-quality event logs for accurate predictions and insights. This research explores the potential of Large Language Models (LLMs) to address challenges in data preparation, including missing values, synonym transformations, and homonym ambiguity. The repository demonstrates how LLM-based imputation enhances data quality, resulting in improved PPM performance.

---

#### **Key Features**
- **LLM-Based Data Imputation:** Python scripts for restoring missing values and handling semantic variability in event logs using LLMs like GPT or LLaMA.
- **Synonym and Homonym Transformation Tools:** Utilities for generating transformed datasets to simulate real-world challenges.
- **PPM Model Evaluation:** End-to-end pipeline to train and evaluate PPM models using both classic and LLM-enhanced datasets.
- **Performance Metrics and Analysis:** Detailed metrics (accuracy, F1-score) and visualizations comparing LLM-based and traditional methods.

---

#### **Repository Structure**
```plaintext
📂 datasets/
    ├── credit/        # Original and transformed event logs for the Credit dataset
    ├── pub/           # Original and transformed event logs for the Pub dataset

📂 models/
    ├── llm/           # Scripts and checkpoints for fine-tuning LLMs
    ├── traditional/   # Scripts for rule-based and statistical imputation methods

📂 experiments/
    ├── preprocessing/ # Synonym and homonym transformation scripts
    ├── training/      # Code for training PPM models on restored datasets
    ├── evaluation/    # Evaluation scripts and metrics visualization

📂 results/
    ├── tables/        # Tabulated metrics for each experiment
    ├── figures/       # Visualizations (accuracy, F1-score trends)

📂 thesis/
    ├── latex/         # LaTeX source files for the dissertation
    ├── figures/       # Diagrams and images used in the thesis
    ├── bibliography/  # References and citations
```

---

#### **How to Use** (It should be revised. !!!!!!!!!!!!)
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Yeoonsu/dissertation.git
   cd dissertation
   ```
   
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Experiments**:
   - Generate transformed datasets:
     ```bash
     python experiments/preprocessing/transform_data.py --dataset credit --transformation synonym
     ```
   - Train a PPM model:
     ```bash
     python experiments/training/train_ppm.py --dataset credit --method llm
     ```
   - Evaluate performance:
     ```bash
     python experiments/evaluation/evaluate_model.py --dataset credit --method llm
     ```

4. **Visualize Results**:
   - Accuracy and F1-score plots:
     ```bash
     python experiments/evaluation/plot_results.py --output results/figures/
     ```

---

#### **Highlights**
- **Reproducibility:** All experiments are fully documented and reproducible.
- **Data Diversity:** Includes two datasets (Credit and Pub) with varied linguistic challenges.
- **Open Access:** Researchers can adapt the pipeline for domain-specific PPM tasks.

---

#### **Citation**
If you use this repository, please cite:

```bibtex
@phdthesis{yeonsu2024thesis,
  title     = {Optimized Data Preparation Pipelines for Predictive Process Monitoring Using LLMs},
  author    = {Yeonsu Kim},
  year      = {2024},
  school    = {UNIST},
  url       = {https://github.com/Yeoonsu/dissertation}
}
```

---

#### **License**
This project is licensed under the [MIT License](LICENSE).

---

#### **Contact**
For questions or collaborations, please contact:  
[Your Name] - *yeon17@unist.ac.kr*  
[GitHub Profile](https://github.com/Yeoonsu)  

--- 
