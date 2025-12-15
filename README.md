PRICING RECOMMENDATION SYSTEM

Project Description
This project focuses on developing a pricing recommendation system for ride-sharing platforms using historical data and machine learning techniques. The system analyzes demand, supply, and customer behavior to recommend optimal ride prices dynamically.

Key Features
- Machine learning–based pricing recommendations
- Dynamic pricing adjustments based on demand and supply
- Customer behavior and loyalty analysis

Dataset Information
Dataset Used: Dynamic Pricing Dataset

Key Features:
- Number of riders and available drivers
- Area type (urban, suburban, rural)
- Customer loyalty status
- Booking time and vehicle category
- Historical pricing data

Installation and Setup

1. Clone the repository:
git clone https://github.com/imane0x/Dynamic-Pricing

2. Install the required dependencies:
pip install -r requirements.txt

3. Train the model:
python main.py

4. Build and run the Docker container:
docker build -t dynamic-pricing .
docker run -p 8000:8000 dynamic-pricing

Model Optimization
Hyperparameter tuning is performed using Grid Search to optimize the Random Forest Regressor and improve prediction accuracy.

Experiment Tracking
Weights & Biases (wandb) is integrated to track experiments, model performance, and training metrics.
