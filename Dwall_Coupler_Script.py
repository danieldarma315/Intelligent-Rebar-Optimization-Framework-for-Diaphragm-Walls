# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 14:03:07 2025

@author: user
"""

import numpy as np
import random
import pandas as pd
import math
from prettytable import PrettyTable

# File path
file_path = r"C:/Users/user/Desktop/Dwall Coupler Case.xlsx"

# Step 1: Read both sheets
sheet1 = pd.read_excel(file_path, sheet_name=0)  # First sheet
sheet2 = pd.read_excel(file_path, sheet_name=1)  # Second sheet

# Step 2: Create a library from Sheet 1
sheet1['Description'] = sheet1['Description'].str.strip().str.lower()  # Normalize descriptions
sheet1_library = dict(zip(sheet1['Description'], sheet1['Contents']))  # Create dictionary

# Debug: Display the library
print("Main Continuous Rebar Optimization")
print("Library from Rebar Specifications:")
for key, value in sheet1_library.items():
    print(f"{key}: {value}")

# Step 3: Retrieve Maximum Rebar Length from the library
max_rebar_length = sheet1_library.get('maximum rebar length', 12000.0)  # Default to 12000.0 if missing

if pd.isna(max_rebar_length):
    print("Warning: 'Maximum Rebar Length' is missing or NaN. Using default value of 12000.0 mm.")
    max_rebar_length = 12000.0

print(f"\nMaximum Rebar Length: {max_rebar_length:.1f} mm")

# Step 4: Calculate total lengths and number of special length rebars for all sections
sheet2 = sheet2.fillna(0)  # Replace NaN with 0

layer = sheet2['Layer']  # List of sections
diameter = sheet2['Diameter']
number_of_rebars_in_bundle = sheet2['No of rebar in bundle']
total_bar_length = sheet2['Total bar length']

# Dictionaries to store calculations
total_length_library = {}
number_of_special_length = {}

for i in range(len(layer)):
    total_length_library[i] = total_bar_length[i]
    number_of_special_length[i] = math.ceil(total_bar_length[i] / max_rebar_length)

# Display results
print("\nTotal Rebar Lengths and Number of Special Length Rebars:")
for i in range(len(layer)):
    print(f"Layer {layer[i]}: Length = {total_length_library[i]:.2f} mm, "
          f"Special Rebars = {number_of_special_length[i]}")

# Step 5: Calculate required and purchasable special lengths (with layer-specific gap)
required_special_length = {}
purchasable_special_length = {}
end_bar_length = {}
middle_bar_length = {}

for i in range(len(layer)):
    # Determine gap based on diameter
    dia = diameter[i]
    gap_key = 'inner gap of coupler >h32' if dia > 32 else 'inner gap of coupler <h32'
    gap = sheet1_library.get(gap_key, 20)
    
    # Compute end and middle bar lengths using the adjusted gap
    avg_bar_length = total_length_library[i] / number_of_special_length[i]
    
    # Apply condition when number_of_special_length = 2
    if number_of_special_length[i] == 2:
        end_bar_length[i] = avg_bar_length - (gap / 2)
        middle_bar_length[i] = avg_bar_length - (gap / 2)
    else:
        end_bar_length[i] = avg_bar_length - (gap / 2)
        middle_bar_length[i] = avg_bar_length - gap
    
    # Take the maximum as the required special length
    required_special_length[i] = max(end_bar_length[i], middle_bar_length[i])
    
    # Round up to the nearest 100 mm for purchasable length
    purchasable_special_length[i] = math.ceil(required_special_length[i] / 100) * 100

# Display the results
print("\nRequired and Purchasable Special Lengths with End and Middle Bar Lengths:")
for i in range(len(layer)):
    print(
        f"Layer {layer[i]}: "
        f"End Bar Length = {end_bar_length[i]:.2f} mm, "
        f"Middle Bar Length = {middle_bar_length[i]:.2f} mm, "
        f"Required Length = {required_special_length[i]:.2f} mm, "
        f"Purchasable Length = {purchasable_special_length[i]:.2f} mm"
    )

# Step 6: Calculate required and purchasable weights for each section

# Convert end and middle bar lengths to meters
end_bar_length_m = {i: end_bar_length[i] / 1000 for i in range(len(layer))}
middle_bar_length_m = {i: middle_bar_length[i] / 1000 for i in range(len(layer))}

# Initialize
total_number_of_special_lengths = {}
required_weight = {}
purchasable_weight = {}
number_of_coupler = {}

for i in range(len(layer)):
    dia = int(diameter[i])  # Ensure it's integer like 25, 40, etc.
    unit_weight_key = f"rebar unit weight h{dia}".lower()
    rebar_unit_weight = sheet1_library.get(unit_weight_key)

    if rebar_unit_weight is None:
        print(f"Warning: Missing unit weight for H{dia}. Assigning manually based on diameter.")
        if dia == 40:
            rebar_unit_weight = 9.854
        elif dia == 25:
            rebar_unit_weight = 3.854

    rebars_in_bundle = sheet2.loc[i, 'No of rebar in bundle']
    if pd.isna(rebars_in_bundle) or rebars_in_bundle == 0:
        print(f"Warning: Missing or invalid 'Number of rebars in bundle' for section {layer[i]}. Defaulting to 1.")
        rebars_in_bundle = 1

    total_number_of_special_lengths[i] = number_of_special_length[i] * rebars_in_bundle

    total_end_bars = 2 * rebars_in_bundle if total_number_of_special_lengths[i] > 2 else 2
    total_middle_bars = total_number_of_special_lengths[i] - total_end_bars

    end_bar_quantity = total_end_bars * rebar_unit_weight * end_bar_length_m[i]
    middle_bar_quantity = total_middle_bars * rebar_unit_weight * middle_bar_length_m[i]
    required_weight[i] = end_bar_quantity + middle_bar_quantity

    purchasable_weight[i] = total_number_of_special_lengths[i] * rebar_unit_weight * (
        purchasable_special_length[i] / 1000)
    number_of_coupler[i] =  (number_of_special_length[i]-1) * rebars_in_bundle

# Summary Table
summary_table = PrettyTable()
summary_table.field_names = [
    "Layer",
    "Diameter",
    "Number of Couplers",
    "Purchasable Special Length (mm)",
    "Total Special Length Rebars",
    "Total Required Quantity (kg)",
    "Total Purchased Quantity (kg)",
]

summary_data = []
total_special_rebar = 0
total_required_weight = 0
total_purchasable_weight = 0
total_number_of_couplers = 0  # renamed

for i in range(len(layer)):
    summary_table.add_row([
        layer[i],
        diameter[i],
        number_of_coupler[i],  # correct key
        purchasable_special_length[i],
        total_number_of_special_lengths[i],
        round(required_weight[i], 2),
        round(purchasable_weight[i], 2),
    ])
    
    summary_data.append([
        layer[i],
        diameter[i],
        number_of_coupler[i],
        purchasable_special_length[i],
        total_number_of_special_lengths[i],
        round(required_weight[i], 2),
        round(purchasable_weight[i], 2),
    ])
    
    total_special_rebar += total_number_of_special_lengths[i]
    total_required_weight += required_weight[i]
    total_purchasable_weight += purchasable_weight[i]
    total_number_of_couplers += number_of_coupler[i]  # accumulate total

# Add total row
summary_table.add_row([
    "Total",
    "-",
    total_number_of_couplers,
    "-",
    total_special_rebar,
    round(total_required_weight, 2),
    round(total_purchasable_weight, 2),
])

print("\nRebar Summary Table:")
print(summary_table)

waste_rate = (total_purchasable_weight - total_required_weight) / total_purchasable_weight
waste_rate_percentage = waste_rate * 100
print(f"\nEstimated Rebar Waste Rate: {waste_rate_percentage:.2f}%")

output_file_summary = r"C:/Users/user/Desktop/Rebar_Summary_Dwall_Coupler.xlsx"

summary_df = pd.DataFrame(summary_data, columns=[
   "Layer",
   "Diameter",
   "Number of Couplers",
   "Purchasable Special Length (mm)",
   "Total Special Length Rebars",
   "Total Required Quantity (kg)",
   "Total Purchased Quantity (kg)",
])

# Append total row to DataFrame
summary_df.loc[len(summary_df)] = [
    "Total",      # Layer
    "-",          # Diameter
    total_number_of_couplers, # Number of Couplers
    "-",          # Purchasable Special Length (mm)
    total_special_rebar,
    round(total_required_weight, 2),
    round(total_purchasable_weight, 2),
]
# Append waste rate row
summary_df.loc[len(summary_df)] = [
   "Waste Rate", "", "", "", "", "", f"{waste_rate_percentage} %"
]


# Export to Excel
output_file_summary = r"C:/Users/user/Desktop/Rebar_Summary_Dwall_Coupler.xlsx"
summary_df.to_excel(output_file_summary, index=False)

print("Rebar summary with waste rate saved to", output_file_summary)
