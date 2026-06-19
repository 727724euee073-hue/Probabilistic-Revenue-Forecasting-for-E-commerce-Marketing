# Probabilistic Revenue Forecasting for E-commerce Marketing

## Overview

Probabilistic Revenue Forecasting for E-commerce Marketing is a production-ready machine learning pipeline designed to forecast future revenue and Return on Ad Spend (ROAS) across multiple advertising platforms, including Google Ads, Meta Ads, and Bing Ads.

Unlike traditional forecasting systems that provide only a single estimate, our solution generates probabilistic forecasts with confidence intervals, enabling marketers to make risk-aware and data-driven decisions.

---

## Key Features

* Revenue Forecasting for 30, 60, and 90 days
* P10, P50, and P90 Prediction Intervals
* Blended ROAS Estimation
* Confidence Score Generation
* Automated Feature Engineering Pipeline
* Multi-Platform Marketing Data Integration
* Offline and Reproducible Execution
* Competition-Compliant One-Command Deployment

---

## Problem Statement

Digital marketers often struggle to accurately predict future campaign revenue due to changing customer behavior, seasonality, and advertising performance fluctuations.

Traditional forecasting approaches provide only a single estimate and fail to quantify uncertainty, leading to suboptimal budget allocation and increased business risk.

Our solution addresses this challenge by generating probabilistic revenue forecasts and confidence intervals, allowing organizations to understand both expected outcomes and associated uncertainty.

---

## Solution Architecture

Advertising Data Sources:

* Google Ads
* Meta Ads
* Bing Ads

Pipeline Flow:

Data Ingestion
→ Data Normalization
→ Revenue Estimation
→ Feature Engineering
→ Machine Learning Forecasting
→ Probabilistic Revenue Prediction
→ ROAS & Confidence Scoring

---

## Repository Structure

├── run.sh
├── requirements.txt
├── data/
├── pickle/
│   └── model.pkl
├── src/
└── README.md

---

## Technology Stack

* Python 3.11
* Pandas
* NumPy
* Scikit-Learn
* Joblib
* Git
* GitHub

---

## Installation

pip install -r requirements.txt

---

## Run the Project

./run.sh ./data ./pickle/model.pkl ./output/predictions.csv

---

## Output

The pipeline generates:

* features.csv
* normalized_data.csv
* feature_metadata.json
* predictions.csv

Sample Output Columns:

* forecast_period
* predicted_revenue
* p10
* p50
* p90
* predicted_roas
* confidence_score

---

## Example Forecast Output

| Forecast Period | Predicted Revenue | P10       | P50       | P90       | ROAS   |
| --------------- | ----------------- | --------- | --------- | --------- | ------ |
| 30 Days         | 390,444           | 378,843   | 390,444   | 409,638   | 5.2367 |
| 60 Days         | 782,200           | 767,806   | 782,200   | 810,692   | 5.2601 |
| 90 Days         | 1,171,643         | 1,155,766 | 1,171,643 | 1,208,119 | 5.2695 |

---

## Reproducibility

* No internet dependency during runtime
* No hardcoded paths
* Fully automated execution from run.sh
* Compatible with automated evaluation pipelines

---

## Team Information

Team Name: Team Vanguard

Team Leader:
Natarajan B

Email:
[727724euee073@skcet.ac.in](mailto:727724euee073@skcet.ac.in)

Institution:
Sri Krishna College of Engineering and Technology, Coimbatore

---

## NetElixir Hackathon Submission

Repository:
https://github.com/727724euee073-hue/Probabilistic-Revenue-Forecasting-for-E-commerce-Marketing

Execution Command:

./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
