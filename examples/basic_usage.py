"""Basic usage example for SAM-Mosaic."""

from sam_mosaic import segment_image

# Simple usage with default settings
result = segment_image(
    input_path="path/to/input.tif",
    output_dir="output/",
    checkpoint="path/to/sam2.1_hiera_large.pt",
    verbose=True
)

print(f"\nResults:")
print(f"  Segments: {result.n_segments}")
print(f"  Coverage: {result.coverage:.1f}%")
print(f"  Time: {result.processing_time:.1f}s")
print(f"  Labels: {result.labels_path}")
print(f"  Shapefile: {result.shapefile_path}")
