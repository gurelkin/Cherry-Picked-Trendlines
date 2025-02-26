# **Detecting Cherry Picked Trendlines - Statistical Analysis of Data-Based Statements**

## **Overview**
This project provides a framework for evaluating statistical statements over structured datasets. It calculates the **"support"** of statements, i.e. how "strong" or "correct" it is, using both brute-force and optimized methods, as well as **Monte Carlo sampling techniques**. The project is built with **Python** and uses **Pandas, NumPy, SciPy, and Red-Black Trees** for efficient calculations.

The framework is structured to analyze **statements of comparative nature** across different datasets (e.g., temperature changes, economic trends, property prices) and to generate both **numerical results and visualizations**.

* The `report.ipynb` contains an extensive mathematical and algorithmic analysis of the algorithms implemented in this project.
* Also, the report contains interesting experiments and explanations on them which we conducted upon real-world datasets.
## **Directory Structure**
```
├── data/                  # Contains CSV datasets
├── experiments.py         # Defines various statistical experiments
├── support.py             # Core support computation functions
├── rbt.py                 # Red-Black Tree implementation
├── region_extractor.py    # Region querier/filterer
├── report.ipynb           # Jupyter Notebook with experiments explainations, results and graphs
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## **Features**
- **Baseline and Exact Support Calculations**
  - `baseline_constrained`, `exact_constrained`, `baseline_unconstrained`, `exact_unconstrained`
- **Monte Carlo Sampling Methods**
  - `pair_sampling`, `point_sampling`
- **Statement Optimizations**
  - `tightest_statement`, `most_supported_statement`
- **Experiments on Real-World Use-Cases**
  - **Algorithms Analysis**
    - `beer_sheva_temps`, `dead_sea_level`, `african_gdp`, `germany_rainfall`, `danish_house`
  - **Graphical Analysis**
    - `beer_sheva_temps_ts_graph`, `beer_sheva_temps_mss_graph`, `stochastic_danish_house`

## **Installation**
Ensure you have the following dependencies installed:
```bash
pip install -r requirements.txt
```
Additionally, ensure that the `RedBlackTree` implementation is accessible under the `rbt` module.

## **Usage**
The project consists of different types of experiments. You can run them via the `experiments.py` script or interactively using `report.ipynb`.

### **1. Running Experiments**
To execute predefined experiments, use:
```python
import experiments
experiments.beer_sheva_temps()
# ...
```

### **2. Running the Jupyter Report**
To analyze results interactively:
```bash
jupyter notebook report.ipynb
```

## **Experiments Overview**
### **1. Beer-Sheva Temperature Analysis**
- Evaluates whether summer temperatures in Beer-Sheva are higher than winter.
- Uses **unconstrained** statistical comparisons.
- Produces a **graphical representation** of tightest statements per support percentage.

### **2. Dead Sea Level Analysis**
- Compares water levels before and after 1991 to determine if they have increased.
- Uses **unconstrained** statistical comparisons.
- Uses **exact and baseline support computations**.

### **3. African GDP Analysis**
- Assesses whether African countries GDPs were affected by the 2008 financial crisis.
- Includes **constraint-based comparisons** to ensure accurate cross-country GDP trends.

### **4. German Rainfall Analysis**
- Compares rainfall in Berlin vs. Düsseldorf under identical time conditions.
- Uses **constrained** statistical comparisons.
- Uses **tightest and most-supported statement optimizations**.

### **5. Danish Housing Prices (Stochastic Analysis)**
- Randomly samples a subset of 10,000 housing records, due to full data being too large.
- Iteratively increases sample size and computes support values, per sampled subset.
- **Visualizes exact vs. point sampling results**.

## **Graphical Analysis**
### **Tightest Statement per Support**
- **Visualization:** Displays how statement bounds change across different support levels.
- **Function:** `beer_sheva_temps_ts_graph()`

### **Most Supported Statement per Statement Range**
- **Visualization:** Shows the most statistically supported statement and its support percent, per numerical range.
- **Function:** `beer_sheva_temps_mss_graph()`

### **Exact vs. Point Sampling (Danish Housing Market)**
- **Visualization:** Compares accuracy of Monte Carlo **point sampling** vs. **exact unconstrained support**.
- **Function:** `stochastic_danish_house()`

## **Key Algorithms Used**
1. **Brute-Force Computation (Baseline Methods)**
   - Compares all valid region pairs to compute support.
   - O(N²) complexity.

2. **Optimized Computation (Binary Search/Red-Black Tree Methods)**
   - Uses an **ordered array** or **self-balancing Red-Black Tree** to store order statistics.
   - Improves efficiency to **O(N log N)**.

3. **Monte Carlo Sampling (Pair & Point Sampling)**
   - Efficiently estimates support with random sampling.
   - Can provide confidence/error margin for results.
   - O(N) complexity.

## **Contributors**
   - Shahar Blank
   - Gur Elkin

## **References**
This project is a reproduce work of the paper "On Detecting Cherry-picked Trendlines".
   - Abolfazl Asudeh
   - H. V. Jagadish
   - You (Will) Wu
   - Cong Yu

   * https://www.vldb.org/pvldb/vol13/p939-asudeh.pdf
   * https://dl.acm.org/doi/10.14778/3380750.3380762

---
🚀 **Developed for Advanced Data-Driven Statistical Analysis.**

