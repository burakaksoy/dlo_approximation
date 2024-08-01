import pandas as pd

"""
Takes a csv file with a similar content to this:
    ```
    [0]/id,0.000000000
    [0]/o_w,0.706433772
    [0]/o_x,0.706433772
    [0]/o_y,-0.030843565
    [0]/o_z,-0.030843565
    [0]/p_x,0.100000000
    [0]/p_y,0.200000000
    [0]/p_z,0.450000000
    [10]/id,10.000000000
    [10]/o_w,-0.694079024
    [10]/o_x,-0.705812747
    [10]/o_y,0.099778600
    [10]/o_z,0.100632529
    [10]/p_x,0.086784465
    [10]/p_y,0.076003991
    [10]/p_z,0.448744727
    [11]/id,11.000000000
    [11]/o_w,0.681778294
    [11]/o_x,0.694047275
    [11]/o_y,-0.162596097
    [11]/o_z,-0.164436147
    [11]/p_x,0.082282784
    [11]/p_y,0.064302298
    [11]/p_z,0.448537719
    [12]/id,12.000000000
    [12]/o_w,-0.681655276
    [12]/o_x,-0.693927430
    [12]/o_y,0.163098284
    [12]/o_z,0.164953802
    [12]/p_x,0.076620083
    [12]/p_y,0.053087242
    [12]/p_z,0.448333220
    [13]/id,13.000000000
    [13]/o_w,0.658632411
    [13]/o_x,0.670798242
    [13]/o_y,-0.239465005
    [13]/o_z,-0.242671748
    [13]/p_x,0.069793256
    [13]/p_y,0.042544058
    [13]/p_z,0.448135740
    [14]/id,14.000000000
    [14]/o_w,-0.658451265
    [14]/o_x,-0.670616515
    [14]/o_y,0.239957390
    [14]/o_z,0.243178685
    [14]/p_x,0.061712053
    [14]/p_y,0.032885307
    [14]/p_z,0.447951615
    [15]/id,15.000000000
    [15]/o_w,-0.620884952
    [15]/o_x,-0.632341660
    [15]/o_y,0.325152624
    [15]/o_z,0.330032836
    [15]/p_x,0.052473040
    [15]/p_y,0.024341400
    [15]/p_z,0.447786082
    ```
Converts into a format such that the titles are as "id,p_x,p_y,p_z,o_x,o_y,o_z,o_w". 
And ensures that the final CSV file is sorted by the 'id' column before it is saved.

"""

def convert_csv(input_file, output_file):
    # Read the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Initialize an empty dictionary to store data
    data = {'id': [], 'p_x': [], 'p_y': [], 'p_z': [], 'o_x': [], 'o_y': [], 'o_z': [], 'o_w': []}

    # Temporary storage for current row values
    current_row = {}

    for line in lines:
        # Split the line to get the key and value
        key, value = line.strip().split(',')
        # Extract the identifier and field name
        identifier, field = key.split('/')
        identifier = int(identifier.strip('[]'))

        # If the current identifier is not in the current_row, it means we need to start a new row
        if 'id' in current_row and current_row['id'] != identifier:
            # Add the current_row data to the main data dictionary
            for k in data.keys():
                data[k].append(current_row[k])
            # Reset current_row for the new identifier
            current_row = {}
        
        # Update the current_row with the new data
        current_row['id'] = identifier
        current_row[field] = float(value)
    
    # Add the last row data to the main data dictionary
    for k in data.keys():
        data[k].append(current_row[k])

    # Create a DataFrame from the dictionary
    df = pd.DataFrame(data)

    # Sort the DataFrame by 'id' in ascending order
    df.sort_values(by='id', ascending=True, inplace=True)

    # Save the DataFrame to the output file
    df.to_csv(output_file, index=False)

# Example usage
input_file = 'dlo_state_example_2.csv'
output_file = 'dlo_state_example_2_clean.csv'
convert_csv(input_file, output_file)
