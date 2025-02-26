# **Detecting Cherry Picked Trendlines - Statistical Validation of Data-Based Statements**

## **Overview**
This project provides a framework for evaluating statistical statements over structured datasets, as presented by Asudeh *et al.* in their paper "On Detecting Cherry-picked Trendlines" \[1\]. Our implementation of the *support metric* quantifies the "correctness" of a statement, based on a given dataset.
The project is structured to analyze statements of comparative nature across different datasets (e.g., temperature changes, economic trends, property prices) and to generate both **numerical results and visualizations**.

The `report.ipynb` notebook contains an extensive coverage of the paper's algorithms, as well as  interesting experiments (with explanations) which we conducted on real-world datasets \[2\].

## **Directory Structure**
```
├── data/                  # Contains CSV datasets, feel free to add your own and experiment!
├── experiments.py         # Defines various statistical experiments
├── support.py             # Core support computation functions
├── rbt.py                 # Red-Black Tree implementation [3]
├── region_extractor.py    # Utilities for handling dataset 'regions' (see paper)
├── report.ipynb           # Jupyter Notebook with experiment explainations, results and plots
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## **Installation**
Clone this repository and ensure you have all the dependencies installed using:
```bash
pip install -r requirements.txt
```

## **Usage**
Our implementation of the various support calculation algorithms from the paper \[1\] are well-documented and can be used programmatically by importing the `support.py` module.
Additionally, this repository comprises different types of experiments, which could be accessed through the `experiments.py` module, or executed interactively in the `report.ipynb` notebook. You can run them via the `experiments.py` script or interactively using `report.ipynb`.

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
   - $O(n^2)$ complexity.

2. **Optimized Computation (Binary Search/Red-Black Tree Methods)**
   - Uses an **ordered array** or **self-balancing Red-Black Tree** to store order statistics.
   - Improves efficiency to $O(n \log n)$.

3. **Monte Carlo Sampling (Pair & Point Sampling)**
   - Efficiently estimates support with random sampling.
   - Can provide confidence/error margin for results.
   - $O(n)$ or $O(n \log n)$ complexity.

## **Contributors**
   - [Shahar Blank](https://github.com/BlankShahar)
   - [Gur Elkin](https://github.com/gurelkin)

## **References**
\[1\] [Asudeh, A., Jagadish, H.V., Wu, Y. and Yu, C., 2020. On detecting cherry-picked trendlines. _Proceedings of the VLDB Endowment_, _13_(6), pp.939-952.](https://www.vldb.org/pvldb/vol13/p939-asudeh.pdf)

\[2\] Datasets used in our experiments:
- [`africa_gdp.csv`](https://www.kaggle.com/datasets/stealthtechnologies/gdp-growth-of-african-countries/data)
- [`beer-sheva_temps.csv`](https://www.visualcrossing.com/weather-query-builder/)
- [`danish_house.csv`](https://www.kaggle.com/datasets/martinfrederiksen/danish-residential-housing-prices-1992-2024?select=DKHousingPricesSample100k.csv)
- [`dead_sea.csv`](https://data.gov.il/dataset/https-www-data-gov-il-dataset-683/resource/823479b4-4771-43d8-9189-6a2a1dcaaf10)
- [`rainfall_germany.csv`](https://www.kaggle.com/datasets/heidarmirhajisadati/germany-city-rainfall-data)


\[3\] Red-Black Tree implementation adapted from [philgookang's repository](https://github.com/philgookang/algorithm_order_statistic_tree/tree/master)