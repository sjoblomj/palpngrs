# Changelog

All notable changes to this project will be documented in this file.


## [0.3.0] - unreleased

### Changed
- Take palette by slice (`&[[u8; 3]]`) instead of `&Vec<[u8; 3]>`, so callers don't have to materialise a `Vec` to pass in their palette.
- Now caching the palette to avoid looking up colours in the wrong palette if multiple calls are made with different palettes.
- No longer maps opaque colours to the transparent colour at index 0.
- Corrected debug offsets in log lines.
- Validate inputs to `draw_image_to_pixel_buffer` and return errors for mismatched image size, out-of-range palette indices, offsets outside the canvas, or an empty palette, instead of panicking.
- Return an error instead of panicking when image dimensions don't fit in the numeric type chosen by the caller.

### Removed
- Removed the `once_cell` dependency, replacing it with `std::sync::LazyLock` instead.


## [0.2.0] - 2025-05-20

### Added
- Function to return a Greyscale palette.
- This Changelog file.
- GitHub Action to auto-publish to Cargo on commit.

### Changed
- Changed function signature to take in a Palette rather than a path.



## [0.1.1] - 2025-05-16

### Added
- Readme.

### Removed
- Removing out-commented code.



## [0.1.0] - 2025-05-16

### Added
- First version of the library. Can convert between PNGs and Palettized Images.
