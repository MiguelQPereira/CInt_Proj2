import pandas as pd

GENERATE = 0

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

# Example of how to call the function with a CSV file
file_path = 'cities.csv'  # Replace with the actual file path
process_cities_csv(file_path)
