# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 14:18:59 2025

@author: asgasga
"""

import numpy as np
import random
import pandas as pd
import math
from prettytable import PrettyTable
from collections import Counter

#Stage 1. Continuous Rebar Optimization Framework (CROF)
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


#Stage 2. WOA-Based Special-Length Cutting Stock Optimization (WSCSO)
# === Load Input Excel with Multiple Sheets ===
input_path = r"C:/Users/user/Desktop/Multiple sheets.xlsx"
sheets_data = pd.read_excel(input_path, sheet_name=None)

# === Prepare Excel Writer for Output ===
output_path = r"C:/Users/user/Desktop/Optimized_MultiSheet_CuttingPatterns.xlsx"
writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

print("Discontinuous and Short Rebar Optimization")
# === Cutting Function ===
def generate_cutting_patterns(stock_length, lengths, quantities):
    patterns = []
    remaining = quantities.copy()
    while sum(remaining) > 0:
        pattern = []
        rem_length = stock_length
        temp_quantities = remaining.copy()

        for i, length in sorted(enumerate(lengths), key=lambda x: -x[1]):
            while rem_length >= length and temp_quantities[i] > 0:
                pattern.append(length)
                rem_length -= length
                temp_quantities[i] -= 1

        if not pattern:
            break

        patterns.append({
            'Pattern': pattern,
            'Waste': round(stock_length - sum(pattern), 3)
        })

        for length in pattern:
            idx = lengths.index(length)
            remaining[idx] -= 1

    return patterns, remaining

# === Fitness Function ===
def fitness_function(stock_length, required_lengths, required_quantities):
    if any(stock_length < l for l in required_lengths):
        return float('inf')
    patterns, _ = generate_cutting_patterns(stock_length, required_lengths, required_quantities.copy())
    total_waste = sum(p['Waste'] for p in patterns)
    penalty = total_waste * (0.5 + random.uniform(0.1, 0.4))
    return total_waste + penalty

# === Run Optimization for Each Sheet ===
for sheet_name, df in sheets_data.items():
    if 'Length' not in df.columns or 'Quantity' not in df.columns:
        print(f"Skipping invalid sheet: {sheet_name}")
        continue

    required_lengths = df['Length'].round(3).tolist()
    required_quantities = df['Quantity'].tolist()

    # === Whale Optimization Algorithm ===
    num_whales = 30
    num_iterations = 50
    whales = np.random.uniform(6.0, 12.0, num_whales)
    best_whale = whales[0]
    best_fitness = fitness_function(best_whale, required_lengths, required_quantities)

    for iteration in range(num_iterations):
        a = 2 - iteration * (2 / num_iterations)
        for i in range(num_whales):
            r = random.random()
            A = 2 * a * r - a
            C = 2 * r
            l = random.uniform(-1, 1)
            p = random.random()

            if p < 0.5:
                if abs(A) < 1:
                    D = abs(C * best_whale - whales[i])
                    whales[i] = best_whale - A * D
                else:
                    rand_idx = random.randint(0, num_whales - 1)
                    D = abs(C * whales[rand_idx] - whales[i])
                    whales[i] = whales[rand_idx] - A * D
            else:
                D = abs(best_whale - whales[i])
                whales[i] = D * np.exp(l) * np.cos(2 * np.pi * l) + best_whale

            whales[i] = round(np.clip(whales[i], 6.0, 12.0) * 10) / 10
            fitness = fitness_function(whales[i], required_lengths, required_quantities)
            if fitness < best_fitness:
                best_fitness = fitness
                best_whale = whales[i]

    # === Extract Final Results ===
    optimal_stock_length = round(best_whale, 1)
    patterns, _ = generate_cutting_patterns(optimal_stock_length, required_lengths, required_quantities.copy())

    pattern_counter = Counter()
    pattern_waste = {}

    for p in patterns:
        sorted_pattern = tuple(sorted(p['Pattern'], reverse=True))
        pattern_counter[sorted_pattern] += 1
        pattern_waste[sorted_pattern] = p['Waste']

    summarized_patterns = []
    for pattern, count in pattern_counter.items():
        summarized_patterns.append({
            "Cutting Pattern": f"[{', '.join(map(str, pattern))}]",
            "Cut from (m)": optimal_stock_length,
            "Waste (m)": pattern_waste[pattern],
            "Used Count": count
        })

    summary_df = pd.DataFrame(summarized_patterns).sort_values(by="Waste (m)").reset_index(drop=True)

    # === Optimization Overview ===
    total_waste = sum(p['Waste'] for p in patterns)
    total_bars = len(patterns)
    waste_rate = round((total_waste / (total_bars * optimal_stock_length)) * 100, 2)

    result_df = pd.DataFrame({
        "Metric": [
            "Optimal Stock Length (m)",
            "Total Bars Needed",
            "Total Waste (m)",
            "Waste Rate (%)"
        ],
        "Value": [
            optimal_stock_length,
            total_bars,
            round(total_waste, 2),
            f"{waste_rate} %"
        ]
    })

    # === Display on Screen ===
    print(f"\n🔧 Sheet: {sheet_name}")
    print("📊 Optimization Result:")
    print(result_df.to_string(index=False))

    print("\n📦 Cutting Pattern Summary (Sorted by Waste Efficiency):")
    print(summary_df.to_string(index=False))

    # === Prepare summary_df with two columns to match result_df for concatenation ===
    summary_df_simple = pd.DataFrame({
        "Metric / Cutting Pattern": summary_df["Cutting Pattern"],
        "Value / Details": summary_df.apply(
            lambda row: f"Cut from {row['Cut from (m)']} m, Waste {row['Waste (m)']} m, Used {row['Used Count']}",
            axis=1
        )
    })

    # Add an empty row for spacing
    empty_row = pd.DataFrame([["", ""]], columns=["Metric / Cutting Pattern", "Value / Details"])

    # === Combine Both Tables in One Sheet ===
    final_df = pd.concat([result_df.rename(columns={"Metric": "Metric / Cutting Pattern", "Value": "Value / Details"}),
                          empty_row,
                          summary_df_simple], ignore_index=True)

    # === Write to Excel ===
    final_df.to_excel(writer, sheet_name=sheet_name, index=False)

# === Finalize Output ===
writer.close()
print(f"✅ Output saved to: {output_path}")