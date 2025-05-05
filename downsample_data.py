import numpy as np
import os
import math

# --- Configuration ---
INPUT_DIR = "/home/lirz6/PROGRAM/DRPR/AE_LSTM/data"
OUTPUT_DIR = os.path.join(INPUT_DIR, "mini")
FILES_TO_PROCESS = ["salinity", "wind", "runoff3"] # Base names of files
ORIGINAL_LENGTH = 2557
TARGET_LENGTH = 257
# --- End Configuration ---

def downsample_data():
    """
    Loads specified .npy files, downsamples them along the first axis,
    and saves the results to the output directory.
    """
    print(f"Input directory: {os.path.abspath(INPUT_DIR)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")

    # Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Ensured output directory exists: {OUTPUT_DIR}")

    # Calculate the step size for sampling
    # step = math.ceil(ORIGINAL_LENGTH / TARGET_LENGTH)
    # print(f"Calculated step size for sampling: {step}")

    # Generate indices using linspace for more even distribution
    indices = np.linspace(0, ORIGINAL_LENGTH - 1, TARGET_LENGTH, dtype=int)
    print(f"Generated {len(indices)} indices for sampling: {indices[:5]}...{indices[-5:]}")


    for base_name in FILES_TO_PROCESS:
        input_file = os.path.join(INPUT_DIR, f"{base_name}.npy")
        output_file = os.path.join(OUTPUT_DIR, f"{base_name}.npy")

        if os.path.exists(input_file):
            print(f"\nProcessing {input_file}...")
            try:
                # Load the data
                data = np.load(input_file)
                print(f"  Loaded data with shape: {data.shape}")

                # Check if the first dimension matches expected length
                if data.shape[0] != ORIGINAL_LENGTH:
                    print(f"  Warning: First dimension ({data.shape[0]}) does not match expected ORIGINAL_LENGTH ({ORIGINAL_LENGTH}). Skipping downsampling based on indices, attempting simple slice if possible.")
                    # Fallback or specific handling if needed, e.g., slice first TARGET_LENGTH
                    if data.shape[0] >= TARGET_LENGTH:
                         downsampled_data = data[:TARGET_LENGTH]
                    else:
                         print(f"  Error: Cannot downsample data with shape {data.shape} to target length {TARGET_LENGTH}. Skipping file.")
                         continue # Skip this file
                else:
                     # Downsample using the generated indices
                     downsampled_data = data[indices, ...] # Ellipsis handles remaining dimensions
                     # Alternative: simple step sampling
                     # downsampled_data = data[::step, ...]


                print(f"  Downsampled data shape: {downsampled_data.shape}")

                # Save the downsampled data
                np.save(output_file, downsampled_data)
                print(f"  Saved downsampled data to: {output_file}")

            except Exception as e:
                print(f"  Error processing file {input_file}: {e}")
        else:
            print(f"\nFile not found, skipping: {input_file}")

if __name__ == "__main__":
    downsample_data()
    print("\nDownsampling process finished.")
