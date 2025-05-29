# 🛍️ Customer Segmentation with K-Means Clustering

A comprehensive data science project that uses machine learning to segment mall customers based on their demographic and behavioral characteristics using K-Means clustering algorithm.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Analysis Workflow](#-analysis-workflow)
- [Key Findings](#-key-findings)
- [Visualizations](#-visualizations)
- [Model Performance](#-model-performance)
- [Technologies Used](#-technologies-used)
- [File Structure](#-file-structure)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

This project implements customer segmentation using K-Means clustering to help businesses understand their customer base better. By analyzing customer demographics and spending behavior, we identify distinct customer groups that can be targeted with specific marketing strategies.

### Business Problem

- **Challenge**: Understanding diverse customer behaviors and preferences
- **Solution**: Segment customers into meaningful groups based on their characteristics
- **Impact**: Enable targeted marketing campaigns and improve customer satisfaction

## 📊 Dataset

The project uses the **Mall Customer Segmentation Data** containing information about mall customers.

### Dataset Features:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Customer gender (Male/Female)
- **Age**: Customer age
- **Annual Income (k$)**: Annual income in thousands of dollars
- **Spending Score (1-100)**: Score assigned based on customer behavior and spending nature

**Data Source**: [Mall Customer Segmentation Dataset](https://media.githubusercontent.com/media/mayurasandakalum/datasets/main/mall-customer-segmentation-k-means/Mall_Customers.csv)

## ✨ Features

- **Comprehensive EDA**: Detailed exploratory data analysis with multiple visualizations
- **Multiple Clustering Approaches**:
  - Age vs Spending Score clustering
  - Annual Income vs Spending Score clustering
  - Multi-dimensional clustering (all features)
- **Elbow Method**: Optimal cluster number determination using WCSS
- **Interactive Visualizations**: 3D scatter plots using Plotly
- **Model Persistence**: Save and load trained models using pickle
- **Customer Prediction**: Predict cluster for new customers

## 🚀 Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/customer-segmentation-k-means.git
   cd customer-segmentation-k-means
   ```

2. **Install required packages**

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn plotly pickle-mixin
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook code.ipynb
   ```

## 📖 Usage

### Running the Analysis

1. Open `code.ipynb` in Jupyter Notebook
2. Run all cells sequentially to:
   - Load and explore the dataset
   - Perform data preprocessing
   - Generate visualizations
   - Apply K-Means clustering
   - Analyze results

### Making Predictions

```python
# Load the saved model
import pickle as pkl
import numpy as np

with open('customer_clustering_model.pkl', 'rb') as f:
    kmeans = pkl.load(f)

# Predict cluster for new customer [Age, Annual Income, Spending Score]
new_customer = np.array([[25, 50, 65]])
cluster = kmeans.predict(new_customer)
print(f"Customer belongs to cluster: {cluster[0]}")
```

## 🔍 Analysis Workflow

### 1. Data Exploration & Preprocessing

- **Data Loading**: Import dataset from GitHub repository
- **Data Cleaning**: Handle missing values and rename columns
- **Statistical Analysis**: Generate descriptive statistics
- **Data Visualization**: Create distribution plots and correlation analysis

### 2. Exploratory Data Analysis

- **Age Distribution**: Histogram and KDE plots
- **Gender Analysis**: Count plots and violin plots
- **Income & Spending Patterns**: Bar charts and relationship plots
- **Categorical Analysis**: Age groups and spending score categories

### 3. K-Means Clustering

- **Feature Selection**: Choose relevant features for clustering
- **Optimal Clusters**: Use Elbow method to determine best k-value
- **Model Training**: Apply K-Means algorithm
- **Visualization**: 2D and 3D cluster visualizations

### 4. Results & Interpretation

- **Cluster Analysis**: Interpret customer segments
- **Business Insights**: Actionable recommendations
- **Model Persistence**: Save trained model for future use

## 🎯 Key Findings

### Customer Segments Identified:

1. **Young Spenders** (Cluster 0)

   - Age: 18-35 years
   - Income: Low-Medium
   - Spending: High
   - Strategy: Target with trendy, affordable products

2. **High Earners** (Cluster 1)

   - Age: 25-45 years
   - Income: High
   - Spending: Medium-High
   - Strategy: Premium products and exclusive offers

3. **Budget Conscious** (Cluster 2)

   - Age: 30-60 years
   - Income: Medium
   - Spending: Low
   - Strategy: Value deals and discounts

4. **Senior Savers** (Cluster 3)

   - Age: 45+ years
   - Income: Medium-High
   - Spending: Low-Medium
   - Strategy: Quality products with long-term value

5. **Premium Shoppers** (Cluster 4)
   - Age: 25-50 years
   - Income: High
   - Spending: High
   - Strategy: Luxury products and VIP services

## 📈 Visualizations

The project includes various types of visualizations:

- **Distribution Plots**: Age, income, and spending score distributions
- **Correlation Analysis**: Relationship between variables
- **Cluster Visualizations**: 2D and 3D scatter plots
- **Elbow Curves**: Optimal cluster determination
- **Bar Charts**: Categorical data analysis

## 🎯 Model Performance

### Clustering Results:

- **Optimal Clusters**: 5 (determined using Elbow method)
- **Algorithm**: K-Means with k-means++ initialization
- **Features Used**: Age, Annual Income, Spending Score
- **Evaluation**: Within-Cluster Sum of Squares (WCSS)

### Model Validation:

- Clear cluster separation in 2D/3D space
- Meaningful business interpretation of clusters
- Consistent results across different feature combinations

## 🛠️ Technologies Used

- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Static visualizations
- **Seaborn**: Statistical data visualization
- **Plotly**: Interactive 3D visualizations
- **Scikit-learn**: Machine learning algorithms
- **Pickle**: Model serialization
- **Jupyter Notebook**: Interactive development environment

## 📁 File Structure

```
customer-segmentation-k-means/
│
├── code.ipynb                          # Main Jupyter notebook
├── customer_clustering_model.pkl       # Trained K-Means model
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or suggestions, please open an issue or contact:

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset provided by [mayurasandakalum](https://github.com/mayurasandakalum)
- Inspired by retail analytics and customer behavior analysis
- Built with open-source tools and libraries

---

⭐ **Star this repository if you found it helpful!** ⭐
