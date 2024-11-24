# Cryptocurrency Price Analysis with Artificial Intelligence

## Overview
This project leverages machine learning techniques, specifically Long Short-Term Memory (LSTM) and Artificial Neural Networks (ANN), to analyze and predict cryptocurrency prices. By utilizing historical price data and advanced data preprocessing methods, the system aims to provide accurate forecasts that can assist investors in making informed decisions.

## Features
- **Price Prediction**: Predict future cryptocurrency prices using LSTM and ANN models.
- **Data Visualization**: Visualize historical price trends and predictions for better insights.
- **User -Friendly Interface**: Easy-to-navigate web interface for users to interact with the system.
- **Data Management**: Efficient handling of cryptocurrency data using Pandas and NumPy.
- **Performance Evaluation**: Comprehensive evaluation of model accuracy using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) metrics.

## Installation
To set up the project locally, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Cryptocurrency_Price_Analysis.git
   cd Cryptocurrency_Price_Analysis
   ```
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up the database**:
   Ensure you have a MySQL server running and create a database named `cryptbitcoin`. Update the database connection settings in `settings.py`.
5. **Run migrations**:
   ```bash
   python manage.py migrate
   ```
6. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

## Usage
- Access the application by navigating to `http://127.0.0.1:8000` in your web browser.
- Users can register, log in, and start trading cryptocurrencies.
- Admins can manage users, agents, and view current cryptocurrency rates.

## Contributing
Contributions are welcome! If you would like to contribute to this project, please fork the repository and submit a pull request. Ensure to follow the coding standards and include relevant tests for new features.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Special thanks to the contributors and the open-source community for their invaluable resources and support.
- Inspired by the growing interest in cryptocurrencies and the need for predictive analytics in financial markets.