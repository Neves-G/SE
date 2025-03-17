# Dash Energy Consumption Analysis

This project is a Dash application designed to visualize and analyze energy consumption data for the IST Civil Building. It allows users to interact with the data, make predictions using a pre-trained neural network model, and apply custom formulas to the dataset.

## Project Structure

```
dash-app
├── data
│   ├── IST_Civil_Pav_2017_2018.csv
│   └── 2019Test.csv
├── models
│   └── NN_model.pkl
├── src
│   ├── app.py
│   └── assets
│       └── styles.css
├── requirements.txt
└── README.md
```

## Files Description

- **src/app.py**: The main application file for the Dash app. It imports necessary libraries, loads data and models, creates visualizations, and sets up the layout and callbacks for user interaction. It includes functionality for users to input a formula that can be applied to the dataset.

- **data/IST_Civil_Pav_2017_2018.csv**: Contains historical data related to energy consumption for the IST Civil Building for the years 2017 and 2018.

- **data/2019Test.csv**: Contains test data for the year 2019, which is used for predictions and analysis in the Dash app.

- **models/NN_model.pkl**: A pre-trained neural network model saved in a pickle format, used for making predictions based on the input data.

- **src/assets/styles.css**: Contains custom styles for the Dash application, allowing for the customization of the app's appearance.

- **requirements.txt**: Lists the dependencies required for the project, including Dash, Plotly, Pandas, and any other necessary libraries.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dash-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

To run the Dash application, execute the following command in your terminal:
```
python src/app.py
```

The application will be accessible at `http://127.0.0.1:8050/` in your web browser.

## Features

- Visualize energy consumption data through interactive graphs.
- Make predictions using a pre-trained neural network model.
- Input custom formulas to manipulate and analyze the dataset.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you would like to add.