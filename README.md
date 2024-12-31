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
- **PPM Model Evaluation:** End-to-end pipeline to train and evaluate PPM models using both classic and LLM-enhanced datasets. (The PPM model implementation is based on [Imperfection-Pattern](https://github.com/brucks1217/Imperfection-pattern))
- **Performance Metrics and Analysis:** Detailed metrics (accuracy, F1-score) and visualizations comparing LLM-based and traditional methods.

---

#### **Repository Structure**
```plaintext
📂 datasets
📂 models/
    ├── llm           # Scripts and checkpoints for fine-tuning LLMs
📂 ppm                # revised and forked from https://github.com/brucks1217/Imperfection-pattern

```

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
  school    = {Ulsan National Institute of Science and Technology},
  url       = {https://github.com/Yeoonsu/dissertation}
}
```

---

#### **License**
This project is licensed under the [MIT License](LICENSE).

---

#### **Contact**
For questions or collaborations, please contact:  
[Yeonsu Kim] - *yeon17@unist.ac.kr*  
[GitHub Profile](https://github.com/Yeoonsu)  

--- 
