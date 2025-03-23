import pandas as pd

def fix_solution_file(input_file, output_file):
    """
    Fix the test_data_solution.txt file to ensure it has the correct format (id ::: genre).
    """
    # Read the file line by line
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Process each line to extract 'id' and 'genre'
    fixed_data = []
    for line in lines:
        # Split the line into parts
        parts = line.strip().split(' ::: ')
        
        # If the line is already in the correct format, keep it
        if len(parts) == 2:
            fixed_data.append(parts)
        else:
            # Handle malformed lines
            # Example: "1 Edgar's Lunch (1998)              thriller"
            # Extract the first word as 'id' and the last word as 'genre'
            words = line.strip().split()
            if len(words) >= 2:
                id_ = words[0]
                genre = words[-1]
                fixed_data.append([id_, genre])
            else:
                print(f"Skipping malformed line: {line}")

    # Save the fixed data to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        for id_, genre in fixed_data:
            file.write(f"{id_} ::: {genre}\n")

    print(f"Fixed data saved to {output_file}")

if __name__ == "__main__":
    input_file = "data/test_data_solution.txt"  # Path to the original file
    output_file = "data/test_data_solution_fixed.txt"  # Path to the fixed file
    fix_solution_file(input_file, output_file)