# Example Python script for ZEN software

import zen

# Access an image
image = zen.application.documents.open(r"D:\RawDataLaurent\Source\2023-02-01\20230201 x10 brigh 228 position 25Zstack.czi")

# Perform an operation, like measuring intensity
measurement = zen.measurements.measure(image, ["MeanIntensity"])

# Save the results
measurement.save(r"D:\RawDataLaurent\output_ZEN\Results.csv")

# Close the image
image.close()