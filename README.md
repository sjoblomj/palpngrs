# palpngrs
Rust library for converting between Palettized images and PNGs.
Palettized images are images that use an external colour palette
to define their colours. This reduces the size of the images and
is frequently used in older games. Rather than containing RGB
pixels, Palettized images contain indices into a palette.

This library can read 256 RBG palettes and convert Palettized
images to PNGs. It can also convert PNGs to Palettized images
by looking up each pixel's RGB value in the palette.

## Transparency
Palette index `0` is reserved for the transparent colour.
Fully-transparent PNG pixels are written as `0`, and opaque
pixels are never mapped to `0` (even if `palette[0]` is the
closest RGB match). When drawing back to a PNG with transparency
enabled, index `0` becomes a transparent pixel and all other
indices become opaque.

## Building
Requires Rust 1.85 or newer (for the 2024 edition).

```sh
cargo build              # debug build
cargo build --release    # optimised build
cargo test               # run the test suite
```
