# Changelog

All notable changes to this project will be documented in this file.


## [0.3.0] - unreleased
### Added
- `palettized_image_with_metadata_to_png` takes a full `PalettizedImageWithMetadata` and writes it to a PNG sized to the original canvas, so trimmed images produced by `read_png` can be round-tripped in one call.
- `PalettizedImageWithMetadata::new` constructor for use by external callers; the struct is now `#[non_exhaustive]` so future fields can be added without a SemVer break.
- `PalettizedImageWithMetadata` now derives `Debug`, `Clone`, `PartialEq`, `Eq`, and `Hash`, so it can be logged, cloned, and compared in user code without manual impls.
- Two new sections to the readme.

### Changed
- `greyscale_palette` returns the palette directly instead of wrapping it in `Result`, since it cannot fail. Callers that previously used `?` or `unwrap` should drop them.
- Take palette by slice (`&[[u8; 3]]`) instead of `&Vec<[u8; 3]>`, so callers don't have to materialise a `Vec` to pass in their palette.
- `read_png` now uses a single flat `Vec<u8>` with stride indexing instead of a `Vec<Vec<u8>>` scratch buffer, removing the per-row allocations and a redundant full copy.
- `read_png` only converts the decoded image to RGBA when it actually has an alpha channel; RGB inputs are decoded as RGB, avoiding a full-image allocation.
- `read_rgb_palette` now returns `ErrorKind::InvalidData` with a descriptive message when the palette file is not exactly 768 bytes, instead of silently ignoring trailing bytes or surfacing a bare `UnexpectedEof`.
- Replaced the global colour-index cache with a per-call local `HashMap`. Consecutive calls with different palettes no longer give incorrect results.
- Relaxed dependency requirements.
- No longer maps opaque colours to the transparent colour at index 0.
- Corrected debug offsets in log lines.
- Validate inputs to `draw_image_to_pixel_buffer` and return errors for mismatched image size, out-of-range palette indices, offsets outside the canvas, or an empty palette, instead of panicking.
- Validate inputs to `save_rgb_pixels_to_image_file` and return `ErrorKind::InvalidInput` when the pixel buffer length does not match `width * height * channels`, instead of panicking.
- Tightened the bounds on `PalettizedImageWithMetadata` and `read_png` to require `TryInto<u32>` on the offset and size types, matching `draw_image_to_pixel_buffer` so that every constructed value is guaranteed to round-trip through the draw and save paths.
- Path-taking public functions (`read_png`, `palettized_image_to_png`, `palettized_image_with_metadata_to_png`, `read_rgb_palette`, `save_rgb_pixels_to_image_file`) now take `impl AsRef<Path>` instead of `&str`, so callers can pass `Path`, `PathBuf`, or `OsStr` without going through `to_str().unwrap()`.
- `palettized_image_to_png` now takes the palette by slice (`&[[u8; 3]]`) instead of an owned `Vec<[u8; 3]>`, matching the inner APIs and avoiding an unnecessary move.
- `draw_image_to_pixel_buffer` now does its inner-loop index arithmetic in `usize` instead of `u32`, matching the buffer-size calculation and removing a theoretical overflow on pathological canvas sizes.
- Tests now write their fixtures into per-test `tempfile::TempDir` directories instead of fixed filenames in the process CWD, so partial-failure leftovers, cross-test filename collisions, and stale data between runs are no longer possible. Added `tempfile` as a dev-dependency.
- Corrected the `palettized_image_to_png` doc comment: it no longer claims the output is always "RGB" (it is RGBA when `use_transparency = true`) and no longer refers to the palette as a "path".
- `read_png` now emits a single summary `warn!` line per call when any input pixels are mapped non-exactly (reporting the count of unique non-exact colours and the maximum squared distance), instead of one `warn!` per unique non-exact colour. Documented that the nearest-colour metric is plain squared Euclidean on raw sRGB and is not perceptually uniform.
- Documented in `read_png` that the function trusts the dimensions of the input PNG and may allocate proportionally large buffers; callers handling untrusted input should pre-validate dimensions or impose their own size cap.
- Return an error instead of panicking when image dimensions don't fit in the numeric type chosen by the caller.
- `read_png` now requires the palette to have at least two entries (index 0 plus at least one opaque colour) and returns an error otherwise.
- Minor stylistic code fixes.

### Removed
- Removed the `once_cell` dependency, replacing it with `std::sync::LazyLock` instead. This requires Rust 1.80 or newer.


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
