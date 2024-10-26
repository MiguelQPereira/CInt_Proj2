import pandas as pd
import csv
import os

GENERATE = 0
MODE = 3

# Function to read the CSV file, check for duplicates, and sort by country
def process_cities_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Check if the required columns 'City' and 'Country' exist in the file
    if 'city' not in df.columns or 'country' not in df.columns:
        print("The file must contain 'city' and 'country' columns.")
        return
    
    # Check for duplicates based on the 'City' column
    duplicates = df[df.duplicated(subset=['city'], keep=False)]
    if not duplicates.empty:
        print("Duplicate cities found:")
        print(duplicates)
    else:
        print("No duplicate cities found.")
    
    # Count how many unique countries there are
    unique_countries = df['country'].nunique()
    unique_cities = df['city'].nunique()
    print(f"Number of different countries: {unique_countries}")
    print(f"Number of different cities: {unique_cities}")

    # Count cities with plain, train and bus
    count_plane = (df['plane'] == 'y').sum()
    nplane = df[df['plane'] != 'y'][['city', 'country']]
    count_train = (df['train'] == 'y').sum()
    ntrain = df[df['train'] != 'y'][['city', 'country']]
    count_bus = (df['bus'] == 'y').sum()
    nbus = df[df['bus'] != 'y'][['city', 'country']]

    print(f"\nNumber of cities with plane: ", count_plane)
    print("Cities without plane stations:\n", nplane)
    print(f"\nNumber of cities with train: ", count_train)
    print("Cities without train stations:\n", ntrain)
    print(f"\nNumber of cities with bus: ", count_bus)
    print("Cities without bus stations:\n", nbus)
    
    if (GENERATE == 1):
        # Sort the DataFrame by 'Country' column, but keep other columns unchanged
        sorted_df = df.sort_values(by='country')
        
        # Save the sorted data back to a CSV file (optional)
        sorted_df.to_csv('output.csv', index=False)
        
        print("Cities sorted by country and saved as 'output.csv'.")

def generate_matrix_csv(file_path):
    # Read the CSV file containing cities
    df = pd.read_csv(file_path)
    
    # Extract the list of cities
    cities = df['city'].tolist()
    
    # Create an empty matrix with dimensions (number of cities + 1) x (number of cities + 1)
    num_cities = len(cities)
    matrix = [['' for _ in range(num_cities + 1)] for _ in range(num_cities + 1)]
    
    # Fill the first row and first column with city names (headers)
    matrix[0][0] = 'City'
    matrix[0][1:] = cities  # First row headers (skipping the first cell)
    for i in range(1, num_cities + 1):
        matrix[i][0] = cities[i - 1]  # First column headers
    
    # Fill the diagonal with '-' and leave the rest empty (or set as needed)
    for j in range(1, num_cities + 1):
        for i in range(1, num_cities + 1):
            matrix[j][i] = '-'
    
    # Save the matrix to a new CSV file
    matrix_df = pd.DataFrame(matrix)
    matrix_df.to_csv('matrix.csv', index=False, header=False)
    
    print(f"Matrix saved to matrix.csv")

def complete_matriz (file_path):
    # Read the CSV file containing the matrix
    df = pd.read_csv(file_path, header=None)
    # Convert the DataFrame to a matrix (list of lists) for easier manipulation
    matrix = df.values.tolist()
    
    # Get the number of cities (assuming square matrix, excluding header row/column)
    num_cities = len(matrix) - 1  # Exclude the header row and column
    
    # Mirror the lower triangle to the upper triangle
    for i in range(1, num_cities + 1):  # Skip first row (headers)
        for j in range(1, i):  # Only look at the lower triangle where i > j
            matrix[j][i] = matrix[i][j]  # Copy lower half to upper half
    
    # Convert the matrix back to a DataFrame
    mirrored_df = pd.DataFrame(matrix)
    
    # Save the updated matrix to a new CSV file
    mirrored_df.to_csv(file_path, index=False, header=False)
    
    print(f"Matrix with mirrored lower triangle saved to {file_path}")

def order_matrix(file_path):
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
    else:
        # Read the CSV file
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # Read the header
            rows = list(reader)    # Read the remaining rows

        # Check if the file has rows to sort
        if not rows:
            print("The file is empty or has only a header.")
        else:
            # Sort rows by the first column (index 0) alphabetically
            rows.sort(key=lambda x: x[0])

            # Write the sorted data back to the same file
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(header)  # Write the header back
                writer.writerows(rows)   # Write the sorted rows

            print(f"The file '{file_path}' has been sorted and saved.")


def decrease_sparse(file_path):
    
    return



# Example of how to call the function with a CSV file
file_path = 'cities.csv'  # Replace with the actual file path
if (MODE == 0):
    process_cities_csv(file_path)
elif (MODE == 1):
    generate_matrix_csv(file_path)
elif (MODE == 2):
    complete_matriz(file_path)
elif (MODE == 3):
    order_matrix(file_path)
elif (MODE == 4):
    decrease_sparse(file_path)