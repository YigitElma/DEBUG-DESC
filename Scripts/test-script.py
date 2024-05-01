import os

# Get the name of the current script
script_name = os.path.basename(__file__)

# Remove the file extension
script_name_without_extension = os.path.splitext(script_name)[0]

print(script_name_without_extension)