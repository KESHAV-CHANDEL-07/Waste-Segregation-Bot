# Path to the single .txt file containing image paths and labels
txt_file_path = 'C:\\Users\\kesha\\test venv\\Depth Estimation\\one-indexed-files-notrash_test.txt'  # Update this path to your file

# Initialize a list to store (image_path, label) tuples
data = []

# Open and read the .txt file
with open(txt_file_path, 'r') as file:
    lines = file.readlines()

# Check if the file is being read correctly
print(f"Number of lines in file: {len(lines)}")
print("First few lines in file:")
print(lines[:431])  # Show the first 5 lines for inspection

# Process each line (image path and label)
for line in lines:
    # Skip empty lines if there are any
    if line.strip() == "":
        continue
    
    
    try:
        img_path, label = line.strip().split()  
        data.append((img_path, int(label)))  
    except ValueError:
        print(f"Skipping line due to formatting issue: {line}")

# Print the entire data array
print(f"Total number of elements: {len(data)}")
print(f"First few elements of data: {data[:431]}")

