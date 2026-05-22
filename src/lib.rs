use image::{ColorType, DynamicImage, ImageBuffer};
use log::{debug, error, info, warn};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io::{Error, ErrorKind, Read};

/// A palettized image plus the offsets and dimensions needed to place it
/// inside its original canvas.
///
/// Palette index `0` is reserved for the transparent colour. Both
/// [`read_png`] and [`draw_image_to_pixel_buffer`] treat index `0` as
/// "transparent" and never use it for an opaque colour, regardless of what
/// RGB value sits at `palette[0]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct PalettizedImageWithMetadata<O, S>
where
    O: TryFrom<u32> + TryInto<u32>, <O as TryInto<u32>>::Error: Debug, // Offset type
    S: TryFrom<u32> + TryInto<u32>, <S as TryInto<u32>>::Error: Debug, // Image size type
{
    /// x-offset to where the image data starts
    pub x_offset: O,
    /// y-offset to where the image data starts
    pub y_offset: O,
    /// width  of the image data
    pub width:    S,
    /// height of the image data
    pub height:   S,
    /// original width  of the image, before any trimming or offsetting was done
    pub original_width:  S,
    /// original height of the image, before any trimming or offsetting was done
    pub original_height: S,
    /// Palettized image, i.e. every element is an index to an external palette.
    /// This is thus not an RGB pixel. Index `0` denotes a transparent pixel.
    pub palettized_image: Vec<u8>,
}

impl<O, S> PalettizedImageWithMetadata<O, S>
where
    O: TryFrom<u32> + TryInto<u32>, <O as TryInto<u32>>::Error: Debug,
    S: TryFrom<u32> + TryInto<u32>, <S as TryInto<u32>>::Error: Debug,
{
    /// Constructs a `PalettizedImageWithMetadata` from its fields. Use this
    /// instead of struct-literal syntax: the struct is `#[non_exhaustive]`,
    /// so fields may be added in future without a SemVer break.
    pub fn new(
        x_offset: O,
        y_offset: O,
        width:           S,
        height:          S,
        original_width:  S,
        original_height: S,
        palettized_image: Vec<u8>,
    ) -> Self {
        Self {
            x_offset, y_offset,
            width, height,
            original_width, original_height,
            palettized_image,
        }
    }
}

/// Given a palettized image and a palette path, this function
/// will create a PNG RGB image in the specified output_path.
///
/// The image is written at its full size with no offset. To preserve the
/// offsets and original canvas size of an image produced by [`read_png`]
/// with `trim_transparent_pixels = true`, use
/// [`palettized_image_with_metadata_to_png`] instead.
pub fn palettized_image_to_png<T>(
    palettized_image: Vec<u8>,
    output_path: &str,
    palette: Vec<[u8; 3]>,
    use_transparency: bool,
    width:  T,
    height: T,
) -> Result<(), Error>
where
    T: Clone + TryFrom<u32> + TryInto<u32>, <T as TryInto<u32>>::Error: Debug,
{
    let image: PalettizedImageWithMetadata<u8, T> = PalettizedImageWithMetadata {
        x_offset: 0,
        y_offset: 0,
        width:  width.clone(),
        height: height.clone(),
        original_width:  width,
        original_height: height,
        palettized_image,
    };
    palettized_image_with_metadata_to_png(image, output_path, &palette, use_transparency)
}

/// Given a [`PalettizedImageWithMetadata`] and a palette, this function
/// will create a PNG image in the specified output_path sized to the
/// `original_width` × `original_height` canvas, with the palettized data
/// placed at `(x_offset, y_offset)`. This is the round-trip counterpart of
/// [`read_png`] with `trim_transparent_pixels = true`.
pub fn palettized_image_with_metadata_to_png<O, S>(
    image: PalettizedImageWithMetadata<O, S>,
    output_path: &str,
    palette: &[[u8; 3]],
    use_transparency: bool,
) -> Result<(), Error>
where
    O: TryFrom<u32> + TryInto<u32>, <O as TryInto<u32>>::Error: Debug,
    S: Clone + TryFrom<u32> + TryInto<u32>, <S as TryInto<u32>>::Error: Debug,
{
    let canvas_width  = to_u32(image.original_width.clone(),  "original_width")?;
    let canvas_height = to_u32(image.original_height.clone(), "original_height")?;
    let rgb_pixels = draw_image_to_pixel_buffer(image, palette, use_transparency)?;
    save_rgb_pixels_to_image_file(rgb_pixels, output_path, use_transparency, canvas_width, canvas_height)
}


/// Reads a Palette file. Expects exactly 768 bytes (256 RGB entries * 3
/// bytes); returns [`ErrorKind::InvalidData`] if the file is shorter or
/// longer than that.
pub fn read_rgb_palette(pal_path: &str) -> std::io::Result<Vec<[u8; 3]>> {
    const PALETTE_BYTES: usize = 768; // 256 RGB entries * 3 bytes
    let mut file = File::open(pal_path)?;
    let mut buffer = Vec::with_capacity(PALETTE_BYTES);
    file.read_to_end(&mut buffer)?;
    if buffer.len() != PALETTE_BYTES {
        return Err(Error::new(ErrorKind::InvalidData, format!(
            "palette file {} is {} bytes, expected {} (256 RGB entries * 3 bytes)",
            pal_path, buffer.len(), PALETTE_BYTES,
        )));
    }

    Ok(buffer.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
}

/// Returns greyscale palette with 256 entries
pub fn greyscale_palette() -> Vec<[u8; 3]> {
    let mut palette = [[0u8; 3]; 256];
    for (i, rgb) in palette.iter_mut().enumerate() {
        rgb[0] = i as u8;
        rgb[1] = i as u8;
        rgb[2] = i as u8;
    }
    Vec::from(palette)
}


/// Saves the given RGB pixel buffer to the given output path.
pub fn save_rgb_pixels_to_image_file(
    rgb_pixels: Vec<u8>,
    output_path: &str,
    use_transparency: bool,
    width:  u32,
    height: u32,
) -> Result<(), Error> {
    let channels = if use_transparency { 4usize } else { 3usize };
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(channels))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "buffer size overflows usize"))?;
    if rgb_pixels.len() != expected_len {
        return Err(Error::new(ErrorKind::InvalidInput, format!(
            "rgb_pixels has {} bytes, expected {} ({}x{} * {} channels)",
            rgb_pixels.len(), expected_len, width, height, channels,
        )));
    }

    let image = if use_transparency {
        DynamicImage::ImageRgba8(
            ImageBuffer::from_raw(width, height, rgb_pixels)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "failed to create RGBA image"))?,
        )
    } else {
        DynamicImage::ImageRgb8(
            ImageBuffer::from_raw(width, height, rgb_pixels)
                .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "failed to create RGB image"))?,
        )
    };
    image.save(output_path).map_err(|e| Error::other(e.to_string()))
}

/// Draws a palettized image into an RGB pixel buffer (Vec<u8>).
/// Uses the given palette for colour lookups.
///
/// When `use_transparency` is `true`, pixels with palette index `0` are
/// written with alpha `0` (transparent) and all other indices with alpha
/// `255` (opaque). Palette index `0` is reserved for the transparent
/// colour; see [`PalettizedImageWithMetadata`].
pub fn draw_image_to_pixel_buffer<O, S>(
    image: PalettizedImageWithMetadata<O, S>,
    palette: &[[u8; 3]],
    use_transparency: bool,
) -> std::io::Result<Vec<u8>>
where
    O: TryFrom<u32> + TryInto<u32>, <O as TryInto<u32>>::Error: Debug,
    S: TryFrom<u32> + TryInto<u32>, <S as TryInto<u32>>::Error: Debug,
{
    let height     = to_u32(image.height,          "height")?;
    let width      = to_u32(image.width,           "width")?;
    let x_offset   = to_u32(image.x_offset,        "x_offset")?;
    let y_offset   = to_u32(image.y_offset,        "y_offset")?;
    let max_width  = to_u32(image.original_width,  "original_width")?;
    let max_height = to_u32(image.original_height, "original_height")?;

    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "image dimensions overflow usize"))?;
    if image.palettized_image.len() < expected_len {
        return Err(Error::new(ErrorKind::InvalidInput, format!(
            "palettized_image has {} entries, expected at least {} ({}x{})",
            image.palettized_image.len(), expected_len, width, height,
        )));
    }

    if x_offset.checked_add(width).is_none_or(|v| v > max_width) {
        return Err(Error::new(ErrorKind::InvalidInput, format!(
            "x_offset ({}) + width ({}) exceeds original_width ({})",
            x_offset, width, max_width,
        )));
    }
    if y_offset.checked_add(height).is_none_or(|v| v > max_height) {
        return Err(Error::new(ErrorKind::InvalidInput, format!(
            "y_offset ({}) + height ({}) exceeds original_height ({})",
            y_offset, height, max_height,
        )));
    }

    let channels = if use_transparency { 4usize } else { 3usize };
    let buffer_size = (max_width as usize)
        .checked_mul(max_height as usize)
        .and_then(|v| v.checked_mul(channels))
        .ok_or_else(|| Error::new(ErrorKind::InvalidInput, "buffer size overflows usize"))?;

    if expected_len > 0 {
        if palette.is_empty() {
            return Err(Error::new(ErrorKind::InvalidInput, "palette is empty"));
        }
        let max_index = *image.palettized_image[..expected_len].iter().max().unwrap() as usize;
        if max_index >= palette.len() {
            return Err(Error::new(ErrorKind::InvalidInput, format!(
                "palettized_image contains index {} but palette has only {} entries",
                max_index, palette.len(),
            )));
        }
    }

    let mut buffer = vec![0u8; buffer_size];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let palette_index = image.palettized_image[idx] as usize;
            let colour = palette[palette_index];

            let out_x = x + x_offset;
            let out_y = y + y_offset;
            let pixel_index = (out_y * max_width + out_x) as usize;

            if use_transparency {
                let base = pixel_index * 4;
                let intensity = if palette_index == 0 {
                    0
                } else {
                    255
                };
                buffer[base..base + 4].copy_from_slice(&[colour[0], colour[1], colour[2], intensity]);
            } else {
                let base = pixel_index * 3;
                buffer[base..base + 3].copy_from_slice(&[colour[0], colour[1], colour[2]]);
            }
        }
    }

    Ok(buffer)
}

/// Reads a PNG file and creates an PalettizedImageWithMetadata by doing colour
/// lookups using the given palette. If trim_transparent_pixels is set to true,
/// any rows or columns where all pixels are transparent will be trimmed away,
/// so that only the non-transparent parts of the image remains.
///
/// Palette index `0` is reserved for the transparent colour: fully-transparent
/// input pixels are written as `0`, and opaque input pixels are never mapped
/// to `0` (even if `palette[0]` is the closest RGB match). The palette must
/// therefore contain at least two entries (one for the transparent colour
/// plus at least one opaque colour to match against); otherwise an error of
/// kind [`ErrorKind::InvalidInput`] is returned.
pub fn read_png<O, S>(
    png_file_name: &str,
    palette: &[[u8; 3]],
    trim_transparent_pixels: bool,
) -> std::io::Result<PalettizedImageWithMetadata<O, S>>
where
    O: TryFrom<u32> + TryInto<u32>, <O as TryInto<u32>>::Error: Debug,
    S: TryFrom<u32> + TryInto<u32>, <S as TryInto<u32>>::Error: Debug,
{
    if palette.len() < 2 {
        return Err(Error::new(ErrorKind::InvalidInput, format!(
            "palette must have at least 2 entries (index 0 is reserved for transparency), got {}",
            palette.len(),
        )));
    }
    let img = image::open(png_file_name)
        .map_err(|e| Error::other(e.to_string()))?;
    let has_alpha = matches!(
        img.color(),
        ColorType::Rgba8 | ColorType::La8 | ColorType::Rgba16 | ColorType::La16,
    );

    let (width, height) = (img.width(), img.height());
    info!(
        "Reading image {}. Has alpha channel: {}. Dimensions: 0x{:0>2X} * 0x{:0>2X} ({} * {})",
        png_file_name, has_alpha, width, height, width, height,
    );

    let stride = width as usize;
    let (raw, channels) = if has_alpha {
        (img.to_rgba8().into_raw(), 4usize)
    } else {
        (img.to_rgb8().into_raw(), 3usize)
    };
    let mut pixels = vec![0u8; stride * height as usize];
    let mut cache: HashMap<([u8; 3], Option<u8>), u8> = HashMap::new();
    for (i, chunk) in raw.chunks_exact(channels).enumerate() {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        let alpha = if has_alpha { Some(chunk[3]) } else { None };
        pixels[i] = *cache
            .entry((rgb, alpha))
            .or_insert_with(|| map_colour_to_palette_index(rgb, alpha, palette));
    }

    let (new_width, new_height, trim_left, trim_top) = if trim_transparent_pixels {
        trim_away_transparency(&pixels, width, height)
    } else {
        (width, height, 0, 0)
    };

    if trim_left != 0 || trim_top != 0 || new_width != width || new_height != height {
        let mut trimmed = Vec::with_capacity((new_width as usize) * (new_height as usize));
        for y in trim_top..trim_top + new_height {
            let start = (y as usize) * stride + trim_left as usize;
            trimmed.extend_from_slice(&pixels[start .. start + new_width as usize]);
        }
        pixels = trimmed;
    }

    Ok(PalettizedImageWithMetadata {
        x_offset: cast::<O>(trim_left,  "x_offset")?,
        y_offset: cast::<O>(trim_top,   "y_offset")?,
        width:    cast::<S>(new_width,  "width")?,
        height:   cast::<S>(new_height, "height")?,
        original_width:  cast::<S>(width,  "original_width")?,
        original_height: cast::<S>(height, "original_height")?,
        palettized_image: pixels,
    })
}

/// Maps an RGB(A) pixel to a palette index.
///
/// Palette index 0 is reserved for the transparent colour: fully-transparent
/// pixels are always mapped to 0, and opaque pixels are never mapped to 0
/// (even if `palette[0]` happens to be the closest RGB match). This avoids
/// opaque pixels round-tripping as transparent in `draw_image_to_pixel_buffer`.
///
/// Precondition: `palette.len() >= 2`. Callers must validate this; the
/// nearest-colour search skips index 0, so a palette with fewer than two
/// entries would yield an invalid result.
fn map_colour_to_palette_index(colour: [u8; 3], alpha: Option<u8>, palette: &[[u8; 3]]) -> u8 {
    if alpha == Some(0) {
        return 0; // Transparent
    }
    if alpha != Some(255) && alpha.is_some() {
        warn!(
            "Pixel [{}, {}, {}, {}] is neither fully transparent nor fully opaque. Will drop the alpha channel.",
            colour[0], colour[1], colour[2], alpha.unwrap(),
        );
    }
    let mut best_index = 1;
    let mut best_distance = u32::MAX;

    // Skip index 0: it is reserved for the transparent colour.
    for (i, &pal_colour) in palette.iter().enumerate().skip(1) {
        let dr = colour[0] as i32 - pal_colour[0]  as i32;
        let dg = colour[1] as i32 - pal_colour[1]  as i32;
        let db = colour[2] as i32 - pal_colour[2]  as i32;
        let dist = (dr * dr + dg * dg + db * db) as u32;

        if dist < best_distance {
            best_distance = dist;
            best_index = i;
        }
    }

    if best_distance != 0 {
        warn!(
            "Non-exact colour match for pixel [{}, {}, {}] — using palette index {} (distance = {})",
            colour[0], colour[1], colour[2], best_index, best_distance,
        );
    }

    best_index as u8
}

fn trim_away_transparency(pixels: &[u8], width: u32, height: u32) -> (u32, u32, u32, u32) {
    // Determine how many rows/columns to trim from each edge
    let mut trim_top:    u32 = 0;
    let mut trim_bottom: u32 = 0;
    let mut trim_left:   u32 = 0;
    let mut trim_right:  u32 = 0;
    let stride = width as usize;

    // Top
    for y in 0..height as usize {
        if pixels[y * stride .. y * stride + stride].iter().all(|&p| p == 0) {
            trim_top += 1;
        } else {
            break;
        }
    }

    // Bottom
    for y in (0..height as usize).rev() {
        if pixels[y * stride .. y * stride + stride].iter().all(|&p| p == 0) {
            trim_bottom += 1;
        } else {
            break;
        }
    }

    // Left
    for x in 0..stride {
        if (0..height as usize).all(|y| pixels[y * stride + x] == 0) {
            trim_left += 1;
        } else {
            break;
        }
    }

    // Right
    for x in (0..stride).rev() {
        if (0..height as usize).all(|y| pixels[y * stride + x] == 0) {
            trim_right += 1;
        } else {
            break;
        }
    }
    debug!(
        "Trimming 0x{:0>2X} ({}) rows from top, 0x{:0>2X} ({}) from bottom, \
        0x{:0>2X} ({}) from left, 0x{:0>2X} ({}) from right",
        trim_top, trim_top, trim_bottom, trim_bottom, trim_left, trim_left, trim_right, trim_right,
    );


    // Clamp dimensions
    let new_width = if width > trim_left + trim_right {
        width - trim_left - trim_right
    } else {
        error!("Image is too small to trim. Setting width to 0");
        0
    };
    let new_height = if height > trim_top + trim_bottom {
        height - trim_top - trim_bottom
    } else {
        error!("Image is too small to trim. Setting height to 0");
        0
    };

    debug!(
        "width:  0x{:0>2X} ({}),  new_width: 0x{:0>2X} ({}), x_offset: 0x{:0>2X} ({})",
        width, width, new_width, new_width, trim_left, trim_left,
    );
    debug!(
        "height: 0x{:0>2X} ({}), new_height: 0x{:0>2X} ({}), y_offset: 0x{:0>2X} ({})",
        height, height, new_height, new_height, trim_top, trim_top,
    );

    (new_width, new_height, trim_left, trim_top)
}

fn cast<T: TryFrom<u32>>(value: u32, name: &str) -> Result<T, Error> {
    T::try_from(value).map_err(|_| Error::new(ErrorKind::InvalidInput, format!("{} out of range", name)))
}

fn to_u32<T: TryInto<u32>>(value: T, name: &str) -> Result<u32, Error> {
    value.try_into().map_err(|_| Error::new(ErrorKind::InvalidInput, format!("{} out of u32 range", name)))
}


#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage, Rgba, RgbaImage};
    use std::fs;

    fn save_test_png_rgb(path: &str, colour: [u8; 3], width: u32, height: u32) {
        let mut img = RgbImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgb(colour);
        }
        let _ = fs::remove_file(path); // Remove if it already exists
        img.save(path).unwrap();
    }

    fn save_test_png_rgba(path: &str, colour: [u8; 4], width: u32, height: u32) {
        let mut img = RgbaImage::new(width, height);
        for pixel in img.pixels_mut() {
            *pixel = Rgba(colour);
        }
        let _ = fs::remove_file(path); // Remove if it already exists
        img.save(path).unwrap();
    }


    #[test]
    fn detects_alpha_correctly() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path_rgb = "test_rgb.png";
        save_test_png_rgb(path_rgb, [100, 100, 100], 8, 8);

        let result_rgb: PalettizedImageWithMetadata<u8, u16> = read_png(path_rgb, &palette, true)?;
        for i in 0..result_rgb.palettized_image.len() {
            assert_eq!(result_rgb.palettized_image[i], 100);
        }
        fs::remove_file(path_rgb)?;


        let path_rgba = "test_rgba.png";
        save_test_png_rgba(path_rgba, [100, 100, 100, 255], 8, 8);

        let result_rgba: PalettizedImageWithMetadata<u8, u16> = read_png(path_rgba, &palette, true)?;
        for i in 0..result_rgba.palettized_image.len() {
            assert_eq!(result_rgba.palettized_image[i], 100);
        }
        fs::remove_file(path_rgba)?;
        Ok(())
    }

    #[test]
    fn drops_alpha_channel_if_not_0() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path_rgba = "test_rgba_alpha.png";
        save_test_png_rgba(path_rgba, [100, 100, 100, 71], 8, 8);

        let trimmed_image: PalettizedImageWithMetadata<u8, u8> = read_png(path_rgba, &palette, true)?;
        for i in 0..trimmed_image.palettized_image.len() {
            assert_eq!(trimmed_image.palettized_image[i], 100);
        }
        fs::remove_file(path_rgba)?;
        Ok(())
    }

    #[test]
    fn trims_transparent_rows_and_columns() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path = "test_trim.png";
        let mut img = RgbaImage::new(3, 3);

        // Center is visible, borders are fully transparent
        for y in 0..3 {
            for x in 0..3 {
                let alpha = if x == 1 && y == 1 { 255 } else { 0 };
                img.put_pixel(x, y, Rgba([100, 100, 100, alpha]));
            }
        }
        img.save(path).unwrap();

        let trimmed_image: PalettizedImageWithMetadata<u8, u8> = read_png(path, &palette, true)?;
        assert_eq!(trimmed_image.width,    1);
        assert_eq!(trimmed_image.height,   1);
        assert_eq!(trimmed_image.x_offset, 1);
        assert_eq!(trimmed_image.y_offset, 1);

        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn maps_non_exact_colours() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path = "test_colour.png";
        save_test_png_rgb(path, [100, 100, 101], 1, 1);

        let result: PalettizedImageWithMetadata<u8, u16> = read_png(path, &palette, false)?;

        assert_eq!(result.palettized_image[0], 100); // Closest match
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn whole_image_is_transparent_and_trimmed_away() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path = "test_transparency.png";
        save_test_png_rgba(path, [0, 0, 0, 0], 1, 1); // Fully transparent

        let trimmed_image: PalettizedImageWithMetadata<u8, u16> = read_png(path, &palette, true)?;

        assert_eq!(trimmed_image.palettized_image.len(), 0);
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn whole_image_is_transparent_but_not_trimmed_away() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path = "test_transparency_without_trimming.png";
        save_test_png_rgba(path, [0, 0, 0, 0], 1, 1); // Fully transparent

        let trimmed_image: PalettizedImageWithMetadata<u8, u16> = read_png(path, &palette, false)?;

        assert_eq!(trimmed_image.palettized_image.len(), 1);
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn image_exactly_255x255() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path = "test_image_exactly_255x255.png";
        let mut img = RgbaImage::new(255, 255);
        for pixel in img.pixels_mut() {
            *pixel = Rgba([100, 100, 100, 255]);
        }
        img.save(&path).unwrap();

        let result: PalettizedImageWithMetadata<u8, u8> = read_png(path, &palette, true)?;
        assert_eq!(result.width  + result.x_offset, 255);
        assert_eq!(result.height + result.y_offset, 255);
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn image_just_above_255x255() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path = "test_image_just_above_255x255.png";
        let mut img = RgbaImage::new(256, 256);
        for pixel in img.pixels_mut() {
            *pixel = Rgba([100, 100, 100, 255]);
        }
        img.save(&path).unwrap();

        let result: Result<PalettizedImageWithMetadata<u8, u8>, Error> = read_png(path, &palette, false);
        assert!(result.is_err());
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn image_too_many_transparent_pixels() -> Result<(), Error> {
        let palette = greyscale_palette();
        let path = "test_image_too_many_transparent_pixels.png";

        // 300x1 image where the only visible pixel sits at x=260, so trimming
        // produces trim_left=260. That offset does not fit in the u8 offset
        // type chosen below, so read_png must return an error.
        let mut img = RgbaImage::new(300, 1);
        img.put_pixel(260, 0, Rgba([100, 100, 100, 255]));
        img.save(&path).unwrap();

        let result: Result<PalettizedImageWithMetadata<u8, u16>, Error> = read_png(path, &palette, true);
        assert!(result.is_err());
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn cache_is_keyed_per_palette() -> Result<(), Error> {
        let path = "test_cache_per_palette.png";
        save_test_png_rgb(path, [10, 20, 30], 1, 1);

        // Palette A: the exact colour sits at index 5
        let mut palette_a = vec![[0u8; 3]; 256];
        palette_a[5] = [10, 20, 30];
        // Palette B: the exact colour sits at index 9
        let mut palette_b = vec![[0u8; 3]; 256];
        palette_b[9] = [10, 20, 30];

        let result_a: PalettizedImageWithMetadata<u8, u16> = read_png(path, &palette_a, false)?;
        let result_b: PalettizedImageWithMetadata<u8, u16> = read_png(path, &palette_b, false)?;

        assert_eq!(result_a.palettized_image[0], 5);
        assert_eq!(result_b.palettized_image[0], 9);
        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn opaque_pixel_never_maps_to_index_zero() -> Result<(), Error> {
        let path = "test_opaque_not_zero.png";
        // Opaque white pixel whose closest match would otherwise be palette[0]
        save_test_png_rgba(path, [255, 255, 255, 255], 1, 1);

        // Palette where index 0 is the exact white match, but a different
        // (non-exact) white sits at another index.
        let mut palette = vec![[0u8; 3]; 256];
        palette[0] = [255, 255, 255];
        palette[7] = [254, 254, 254];

        let result: PalettizedImageWithMetadata<u8, u16> = read_png(path, &palette, false)?;
        assert_ne!(result.palettized_image[0], 0,
            "opaque pixel must not be mapped to the reserved transparent index");
        assert_eq!(result.palettized_image[0], 7);
        fs::remove_file(path)?;
        Ok(())
    }

    fn make_image(
        x_offset: u32, y_offset: u32, width: u32, height: u32,
        original_width: u32, original_height: u32, pixels: Vec<u8>,
    ) -> PalettizedImageWithMetadata<u32, u32> {
        PalettizedImageWithMetadata {
            x_offset, y_offset, width, height,
            original_width, original_height,
            palettized_image: pixels,
        }
    }

    #[test]
    fn draw_rejects_palettized_image_shorter_than_width_times_height() {
        let palette = vec![[0u8; 3]; 2];
        // Claims 4x4 but only provides 8 entries.
        let image = make_image(0, 0, 4, 4, 4, 4, vec![0u8; 8]);
        let err = draw_image_to_pixel_buffer(image, &palette, false).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn draw_rejects_offset_outside_original_canvas() {
        let palette = vec![[0u8; 3]; 2];
        // x_offset (3) + width (2) > original_width (4)
        let image = make_image(3, 0, 2, 2, 4, 4, vec![0u8; 4]);
        let err = draw_image_to_pixel_buffer(image, &palette, false).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);

        // y_offset (3) + height (2) > original_height (4)
        let image = make_image(0, 3, 2, 2, 4, 4, vec![0u8; 4]);
        let err = draw_image_to_pixel_buffer(image, &palette, false).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn draw_rejects_palette_index_out_of_range() {
        let palette = vec![[0u8; 3]; 3]; // valid indices: 0..=2
        let image = make_image(0, 0, 2, 2, 2, 2, vec![0, 1, 2, 5]);
        let err = draw_image_to_pixel_buffer(image, &palette, false).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn draw_rejects_empty_palette_with_non_empty_image() {
        let palette: Vec<[u8; 3]> = Vec::new();
        let image = make_image(0, 0, 1, 1, 1, 1, vec![0]);
        let err = draw_image_to_pixel_buffer(image, &palette, false).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn draw_rejects_dimensions_that_do_not_fit_in_u32() {
        let palette = vec![[0u8; 3]; 1];
        let too_big: u64 = u32::MAX as u64 + 1;
        let image: PalettizedImageWithMetadata<u64, u64> = PalettizedImageWithMetadata {
            x_offset: 0, y_offset: 0,
            width:  too_big, height: 1,
            original_width:  too_big, original_height: 1,
            palettized_image: vec![0u8; 0],
        };
        let err = draw_image_to_pixel_buffer(image, &palette, false).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn palettized_image_to_png_rejects_dimensions_that_do_not_fit_in_u32() {
        let palette: Vec<[u8; 3]> = vec![[0u8; 3]; 1];
        let too_big: u64 = u32::MAX as u64 + 1;
        let result = palettized_image_to_png(
            vec![0u8; 0],
            "test_overflow_not_created.png",
            palette,
            false,
            too_big,
            1u64,
        );
        let err = result.unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn palettized_image_with_metadata_to_png_round_trips_trimmed_image() -> Result<(), Error> {
        let palette = greyscale_palette();
        let src_path = "test_round_trip_src.png";
        let dst_path = "test_round_trip_dst.png";

        // 5x5 RGBA image: only the centre pixel is opaque (white)
        let mut img = RgbaImage::new(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                let alpha = if x == 2 && y == 2 { 255 } else { 0 };
                img.put_pixel(x, y, Rgba([255, 255, 255, alpha]));
            }
        }
        let _ = fs::remove_file(src_path);
        img.save(src_path).unwrap();

        let trimmed: PalettizedImageWithMetadata<u8, u16> = read_png(src_path, &palette, true)?;
        assert_eq!(trimmed.width,           1);
        assert_eq!(trimmed.height,          1);
        assert_eq!(trimmed.x_offset,        2);
        assert_eq!(trimmed.y_offset,        2);
        assert_eq!(trimmed.original_width,  5);
        assert_eq!(trimmed.original_height, 5);

        palettized_image_with_metadata_to_png(trimmed, dst_path, &palette, true)?;

        // Re-read the output without trimming; the canvas must still be 5x5
        // with the centre pixel mapped to index 255 (white) and the rest to 0.
        let result: PalettizedImageWithMetadata<u8, u16> = read_png(dst_path, &palette, false)?;
        assert_eq!(result.width,  5);
        assert_eq!(result.height, 5);
        for y in 0..5usize {
            for x in 0..5usize {
                let idx = result.palettized_image[y * 5 + x];
                if x == 2 && y == 2 {
                    assert_eq!(idx, 255, "centre pixel should be opaque white");
                } else {
                    assert_eq!(idx, 0, "({},{}) should be transparent", x, y);
                }
            }
        }

        fs::remove_file(src_path)?;
        fs::remove_file(dst_path)?;
        Ok(())
    }

    #[test]
    fn palettized_image_to_png_round_trips_untrimmed_image() -> Result<(), Error> {
        let palette = greyscale_palette();
        let dst_path = "test_wrapper_round_trip.png";

        // 2x2 palettized image, all pixels mapped to index 100.
        let pixels = vec![100u8; 4];
        palettized_image_to_png(pixels, dst_path, palette.clone(), false, 2u32, 2u32)?;

        let result: PalettizedImageWithMetadata<u8, u16> = read_png(dst_path, &palette, false)?;
        assert_eq!(result.width,    2);
        assert_eq!(result.height,   2);
        assert_eq!(result.x_offset, 0);
        assert_eq!(result.y_offset, 0);
        for i in 0..4 {
            assert_eq!(result.palettized_image[i], 100);
        }

        fs::remove_file(dst_path)?;
        Ok(())
    }

    #[test]
    fn palettized_image_with_metadata_constructor_and_derives() {
        let a: PalettizedImageWithMetadata<u8, u16> =
            PalettizedImageWithMetadata::new(1, 2, 3, 4, 5, 6, vec![7, 8, 9]);
        let b = a.clone();
        assert_eq!(a, b);
        assert_eq!(a.x_offset,        1);
        assert_eq!(a.y_offset,        2);
        assert_eq!(a.width,           3);
        assert_eq!(a.height,          4);
        assert_eq!(a.original_width,  5);
        assert_eq!(a.original_height, 6);
        assert_eq!(a.palettized_image, vec![7, 8, 9]);
        // Debug impl exists (just exercise it; no specific format guaranteed)
        let _ = format!("{:?}", a);
    }

    #[test]
    fn save_rgb_pixels_rejects_mismatched_rgb_buffer() {
        // Claims 2x2 RGB (= 12 bytes) but only supplies 11.
        let err = save_rgb_pixels_to_image_file(
            vec![0u8; 11],
            "test_save_bad_rgb.png",
            false,
            2,
            2,
        ).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn save_rgb_pixels_rejects_mismatched_rgba_buffer() {
        // Claims 2x2 RGBA (= 16 bytes) but supplies 15.
        let err = save_rgb_pixels_to_image_file(
            vec![0u8; 15],
            "test_save_bad_rgba.png",
            true,
            2,
            2,
        ).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn read_rgb_palette_reads_exactly_768_bytes() -> Result<(), Error> {
        let path = "test_palette_ok.pal";
        let mut bytes = vec![0u8; 768];
        for i in 0..256 {
            bytes[i * 3]     = i as u8;
            bytes[i * 3 + 1] = (i as u8).wrapping_add(1);
            bytes[i * 3 + 2] = (i as u8).wrapping_add(2);
        }
        fs::write(path, &bytes)?;

        let palette = read_rgb_palette(path)?;
        assert_eq!(palette.len(), 256);
        assert_eq!(palette[0],   [0, 1, 2]);
        assert_eq!(palette[255], [255, 0, 1]);

        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn read_rgb_palette_rejects_short_file() -> Result<(), Error> {
        let path = "test_palette_short.pal";
        fs::write(path, vec![0u8; 767])?;

        let err = read_rgb_palette(path).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn read_rgb_palette_rejects_long_file() -> Result<(), Error> {
        let path = "test_palette_long.pal";
        fs::write(path, vec![0u8; 769])?;

        let err = read_rgb_palette(path).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidData);

        fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn read_png_rejects_palette_shorter_than_two_entries() -> Result<(), Error> {
        let path = "test_short_palette.png";
        save_test_png_rgb(path, [100, 100, 100], 1, 1);

        let empty: Vec<[u8; 3]> = Vec::new();
        let r0: Result<PalettizedImageWithMetadata<u8, u16>, Error> = read_png(path, &empty, false);
        assert_eq!(r0.err().unwrap().kind(), ErrorKind::InvalidInput);

        let one = vec![[0u8; 3]];
        let r1: Result<PalettizedImageWithMetadata<u8, u16>, Error> = read_png(path, &one, false);
        assert_eq!(r1.err().unwrap().kind(), ErrorKind::InvalidInput);

        fs::remove_file(path)?;
        Ok(())
    }
}
