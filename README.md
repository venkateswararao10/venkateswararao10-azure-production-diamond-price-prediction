
# Diamond - Price - Prediction


## About The Project

The Diamond Price Prediction project is an end-to-end machine learning initiative designed to forecast the prices of diamonds. Diamonds are highly valuable gemstones, and their prices can vary significantly based on various factors. This project aims to leverage machine learning techniques to provide accurate predictions of diamond prices, thereby offering valuable insights to both buyers and sellers in the diamond market.

## About the Data

The dataset The goal is to predict the price of a given diamond (Regression Analysis).

There are 10 independent variables (including id):

 - id: the unique identifier of each diamond
 - carat: Carat (ct.) refers to the unique unit of weight measurement used exclusively to weigh gemstones and diamonds.
 - cut: Quality of Diamond Cut
 - color: Color of Diamond
 - clarity: Diamond clarity is a measure of the purity and rarity of the stone, graded by the visibility of these characteristics under 10-power magnification.
 - depth: The depth of the diamond is its height (in millimetres) measured from the culet (bottom tip) to the table (flat, top surface)
 - table: A diamond's table is the facet which can be seen when the stone is viewed face up.
 - x : Diamond X dimension
 - y: Diamond Y dimension
 - z: Diamond Z dimension

### Target variable: 
- price: Price of the given Diamond.
  

Dataset Source Link : ```https://www.kaggle.com/competitions/playground-series-s3e8/data?select=train.csv```

## Built With

 - Pandas
 - Numpy
 - Scikit-learn
 - Flask
 - Seaborn
 - Matplotlib


## Getting Started

This will help you understand how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

## Installation Steps

### Option 1: Installation from GitHub

Follow these steps to install and set up the project directly from the GitHub repository:

1. **Clone the Repository**
   - Open your terminal or command prompt.
   - Navigate to the directory where you want to install the project.
   - Run the following command to clone the GitHub repository:
     ```
     git clone https://github.com/venkateswararao10/azure-production-diamond-price-prediction.git
     ```

2. **Create a Virtual Environment** (Optional but recommended)
   - It's a good practice to create a virtual environment to manage project dependencies. Run the following command:
     ```
     conda create -p <Environment_Name> python==<python version> -y
     ```

3. **Activate the Virtual Environment** (Optional)
   - Activate the virtual environment based on your operating system:
       ```
       conda activate <Environment_Name>/
       ```

4. **Install Dependencies**
   - Navigate to the project directory:
     ```
     cd [project_directory]
     ```
   - Run the following command to install project dependencies:
     ```
     pip install -r requirements.txt
     ```

5. **Run the Project**
   - Start the project by running the appropriate command.
     ```
     python app.py
     ```

6. **Access the Project**
   - Open a web browser or the appropriate client to access the project.
  



