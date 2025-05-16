# palpngrs
Rust library for converting between Palettized images and PNGs.
Palettized images are images that use an external colour palette
to define their colours. This reduces the size of the images and
is frequently used in older games. Rather than containing RGB
pixels, Palettized images contain indices into a palette.

This library can read 256 RBG palettes and convert Palettized
images to PNGs. It can also convert PNGs to Palettized images
by looking up each pixel's RGB value in the palette.
