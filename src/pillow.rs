//! Pictures made exactly as Pillow makes them.
//!
//! The box rule was measured with pictures that Python cut, enlarged and saved with Pillow, and
//! d3-ultra's boxes move when the same picture is encoded differently: a JPEG without chroma
//! subsampling changed the rule's result on 2 of 60 grids. So the server reproduces Pillow's
//! bilinear resize and libjpeg's baseline encoder byte for byte instead of using its own.

/// An RGB picture, three bytes per pixel.
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl Rgb {
    /// The pixels from `left`, `top` up to but not including `right`, `bottom`.
    pub fn crop(&self, left: usize, top: usize, right: usize, bottom: usize) -> Rgb {
        let (width, height) = (right - left, bottom - top);
        let mut data = Vec::with_capacity(width * height * 3);
        for row in top..bottom {
            let start = (row * self.width + left) * 3;
            data.extend_from_slice(&self.data[start..start + width * 3]);
        }
        Rgb {
            width,
            height,
            data,
        }
    }

    /// `Image.resize((width, height), Image.BILINEAR)`.
    pub fn resize_bilinear(&self, width: usize, height: usize) -> Rgb {
        // Pillow resizes the rows first unless the height shrinks more than twice as much as
        // the width.
        let (shrink_x, shrink_y) = (
            self.width as i64 - width as i64,
            self.height as i64 - height as i64,
        );
        let rows_first = !(shrink_y > 0 && shrink_y > shrink_x * 2);
        let horizontal = |image: &Rgb| {
            if image.width == width {
                return None;
            }
            let taps = coefficients(image.width, width);
            let mut data = vec![0; width * image.height * 3];
            for row in 0..image.height {
                let line = &image.data[row * image.width * 3..(row + 1) * image.width * 3];
                for (x, (first, weights)) in taps.iter().enumerate() {
                    for band in 0..3 {
                        let mut sum = 1 << (PRECISION_BITS - 1);
                        for (index, weight) in weights.iter().enumerate() {
                            sum += i32::from(line[(first + index) * 3 + band]) * weight;
                        }
                        data[(row * width + x) * 3 + band] = clip8(sum);
                    }
                }
            }
            Some(Rgb {
                width,
                height: image.height,
                data,
            })
        };
        let vertical = |image: &Rgb| {
            if image.height == height {
                return None;
            }
            let taps = coefficients(image.height, height);
            let stride = image.width * 3;
            let mut data = vec![0; stride * height];
            for (y, (first, weights)) in taps.iter().enumerate() {
                for offset in 0..stride {
                    let mut sum = 1 << (PRECISION_BITS - 1);
                    for (index, weight) in weights.iter().enumerate() {
                        sum += i32::from(image.data[(first + index) * stride + offset]) * weight;
                    }
                    data[y * stride + offset] = clip8(sum);
                }
            }
            Some(Rgb {
                width: image.width,
                height,
                data,
            })
        };
        let (first, second) = if rows_first {
            let first = horizontal(self);
            let second = vertical(first.as_ref().unwrap_or(self));
            (first, second)
        } else {
            let first = vertical(self);
            let second = horizontal(first.as_ref().unwrap_or(self));
            (first, second)
        };
        second.or(first).unwrap_or_else(|| Rgb {
            width: self.width,
            height: self.height,
            data: self.data.clone(),
        })
    }

    /// `Image.save(file, 'JPEG', quality=90)`: baseline, 4:2:0, the standard tables.
    pub fn jpeg_quality_90(&self) -> Vec<u8> {
        encode_jpeg(self)
    }
}

// Pillow's Resample.c, 8 bits per channel.
const PRECISION_BITS: u32 = 32 - 8 - 2;

fn clip8(sum: i32) -> u8 {
    (sum >> PRECISION_BITS).clamp(0, 255) as u8
}

/// For each output position, the first input position it reads and its fixed-point weights.
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.0);
    // The bilinear filter reaches one pixel either side at full size.
    let support = filter_scale;
    (0..output)
        .map(|position| {
            let center = (position as f64 + 0.5) * scale;
            let first = ((center - support + 0.5) as i64).max(0);
            let last = ((center + support + 0.5) as i64).min(input as i64);
            let weights: Vec<f64> = (first..last)
                .map(|x| {
                    let distance = ((x as f64 - center + 0.5) * (1.0 / filter_scale)).abs();
                    if distance < 1.0 { 1.0 - distance } else { 0.0 }
                })
                .collect();
            let total: f64 = weights.iter().sum();
            let fixed = weights
                .iter()
                .map(|&weight| {
                    let weight = if total != 0.0 { weight / total } else { weight };
                    let scaled = weight * f64::from(1 << PRECISION_BITS);
                    (if weight < 0.0 {
                        scaled - 0.5
                    } else {
                        scaled + 0.5
                    }) as i32
                })
                .collect();
            (first as usize, fixed)
        })
        .collect()
}

// libjpeg's baseline encoder, as Pillow drives it: quality 90, 2 × 2 luma sampling, the slow
// integer DCT, the standard Huffman tables.

#[rustfmt::skip]
const NATURAL_ORDER: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10, 17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
#[rustfmt::skip]
const LUMA_QUANT: [u32; 64] = [
    16, 11, 10, 16,  24,  40,  51,  61,
    12, 12, 14, 19,  26,  58,  60,  55,
    14, 13, 16, 24,  40,  57,  69,  56,
    14, 17, 22, 29,  51,  87,  80,  62,
    18, 22, 37, 56,  68, 109, 103,  77,
    24, 35, 55, 64,  81, 104, 113,  92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103,  99,
];
#[rustfmt::skip]
const CHROMA_QUANT: [u32; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];
/// `jpeg_quality_scaling(90)`.
const QUALITY_SCALE: u32 = 20;

// Code counts per length, then the symbols, for the four standard tables.
const DC_LUMA: ([u8; 16], &[u8]) = (
    [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
);
const DC_CHROMA: ([u8; 16], &[u8]) = (
    [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
);
#[rustfmt::skip]
const AC_LUMA: ([u8; 16], &[u8]) = (
    [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d],
    &[
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ],
);
#[rustfmt::skip]
const AC_CHROMA: ([u8; 16], &[u8]) = (
    [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
    &[
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
    ],
);

fn quant_table(base: &[u32; 64]) -> [u16; 64] {
    base.map(|value| ((value * QUALITY_SCALE + 50) / 100).clamp(1, 255) as u16)
}

/// libjpeg-turbo divides by multiplying with a reciprocal. Its three numbers per divisor.
struct Divisor {
    reciprocal: u32,
    correction: u32,
    shift: u32,
}

fn divisor(quant: u16) -> Divisor {
    // The DCT output is eight times larger than the coefficients.
    let divisor = u32::from(quant) << 3;
    let bits = 32 - divisor.leading_zeros() - 1;
    let mut shift = 16 + bits;
    let mut reciprocal = (1u32 << shift) / divisor;
    let remainder = (1u32 << shift) % divisor;
    let mut correction = divisor / 2;
    if remainder == 0 {
        reciprocal >>= 1;
        shift -= 1;
    } else if remainder <= divisor / 2 {
        correction += 1;
    } else {
        reciprocal += 1;
    }
    Divisor {
        reciprocal: reciprocal & 0xffff,
        correction: correction & 0xffff,
        shift,
    }
}

/// jfdctint.c: the slow, accurate integer DCT on samples centred on zero.
fn forward_dct(data: &mut [i32; 64]) {
    const CONST_BITS: u32 = 13;
    const PASS1_BITS: u32 = 2;
    let descale = |value: i32, bits: u32| (value + (1 << (bits - 1))) >> bits;
    let odd = |t4: i32, t5: i32, t6: i32, t7: i32| {
        let (z1, z2, z3, z4) = (t4 + t7, t5 + t6, t4 + t6, t5 + t7);
        let z5 = (z3 + z4) * 9633;
        let (t4, t5, t6, t7) = (t4 * 2446, t5 * 16819, t6 * 25172, t7 * 12299);
        let (z1, z2) = (z1 * -7373, z2 * -20995);
        let (z3, z4) = (z3 * -16069 + z5, z4 * -3196 + z5);
        // Outputs 7, 5, 3, 1 before scaling.
        [t4 + z1 + z3, t5 + z2 + z4, t6 + z2 + z3, t7 + z1 + z4]
    };
    for row in data.as_chunks_mut::<8>().0 {
        let (t0, t7) = (row[0] + row[7], row[0] - row[7]);
        let (t1, t6) = (row[1] + row[6], row[1] - row[6]);
        let (t2, t5) = (row[2] + row[5], row[2] - row[5]);
        let (t3, t4) = (row[3] + row[4], row[3] - row[4]);
        let (t10, t13, t11, t12) = (t0 + t3, t0 - t3, t1 + t2, t1 - t2);
        row[0] = (t10 + t11) << PASS1_BITS;
        row[4] = (t10 - t11) << PASS1_BITS;
        let z1 = (t12 + t13) * 4433;
        row[2] = descale(z1 + t13 * 6270, CONST_BITS - PASS1_BITS);
        row[6] = descale(z1 + t12 * -15137, CONST_BITS - PASS1_BITS);
        let [o7, o5, o3, o1] = odd(t4, t5, t6, t7);
        row[7] = descale(o7, CONST_BITS - PASS1_BITS);
        row[5] = descale(o5, CONST_BITS - PASS1_BITS);
        row[3] = descale(o3, CONST_BITS - PASS1_BITS);
        row[1] = descale(o1, CONST_BITS - PASS1_BITS);
    }
    for column in 0..8 {
        let at = |row: usize| data[row * 8 + column];
        let (t0, t7) = (at(0) + at(7), at(0) - at(7));
        let (t1, t6) = (at(1) + at(6), at(1) - at(6));
        let (t2, t5) = (at(2) + at(5), at(2) - at(5));
        let (t3, t4) = (at(3) + at(4), at(3) - at(4));
        let (t10, t13, t11, t12) = (t0 + t3, t0 - t3, t1 + t2, t1 - t2);
        data[column] = descale(t10 + t11, PASS1_BITS);
        data[4 * 8 + column] = descale(t10 - t11, PASS1_BITS);
        let z1 = (t12 + t13) * 4433;
        data[2 * 8 + column] = descale(z1 + t13 * 6270, CONST_BITS + PASS1_BITS);
        data[6 * 8 + column] = descale(z1 + t12 * -15137, CONST_BITS + PASS1_BITS);
        let [o7, o5, o3, o1] = odd(t4, t5, t6, t7);
        data[7 * 8 + column] = descale(o7, CONST_BITS + PASS1_BITS);
        data[5 * 8 + column] = descale(o5, CONST_BITS + PASS1_BITS);
        data[3 * 8 + column] = descale(o3, CONST_BITS + PASS1_BITS);
        data[8 + column] = descale(o1, CONST_BITS + PASS1_BITS);
    }
}

/// One colour plane, padded to whole blocks.
struct Plane {
    width: usize,
    samples: Vec<u8>,
}

impl Plane {
    /// The quantized coefficients of the block at block column `x`, block row `y`.
    fn block(&self, x: usize, y: usize, divisors: &[Divisor; 64]) -> [i16; 64] {
        let mut data = [0i32; 64];
        for row in 0..8 {
            let start = (y * 8 + row) * self.width + x * 8;
            for column in 0..8 {
                data[row * 8 + column] = i32::from(self.samples[start + column]) - 128;
            }
        }
        forward_dct(&mut data);
        let mut block = [0i16; 64];
        for (index, &value) in data.iter().enumerate() {
            let divisor = &divisors[index];
            // The DCT's results are 16-bit, so this multiplication cannot overflow.
            let quantized = ((value.unsigned_abs() + divisor.correction) * divisor.reciprocal)
                >> (divisor.shift);
            block[index] = if value < 0 {
                -(quantized as i16)
            } else {
                quantized as i16
            };
        }
        block
    }
}

struct Huffman {
    code: [u16; 256],
    size: [u8; 256],
}

impl Huffman {
    fn new(table: &([u8; 16], &[u8])) -> Self {
        let (mut code, mut size) = ([0u16; 256], [0u8; 256]);
        let (mut next, mut symbol) = (0u16, 0);
        for (length, &count) in table.0.iter().enumerate() {
            for _ in 0..count {
                let value = usize::from(table.1[symbol]);
                code[value] = next;
                size[value] = length as u8 + 1;
                next += 1;
                symbol += 1;
            }
            next <<= 1;
        }
        Self { code, size }
    }
}

#[derive(Default)]
struct Bits {
    out: Vec<u8>,
    buffer: u32,
    held: u32,
}

impl Bits {
    fn put(&mut self, code: u32, size: u32) {
        if size == 0 {
            return;
        }
        self.buffer = (self.buffer << size) | (code & ((1 << size) - 1));
        self.held += size;
        while self.held >= 8 {
            let byte = (self.buffer >> (self.held - 8)) as u8;
            self.out.push(byte);
            if byte == 0xff {
                self.out.push(0);
            }
            self.held -= 8;
        }
        self.buffer &= (1 << self.held) - 1;
    }

    fn block(&mut self, block: &[i16; 64], previous_dc: &mut i16, dc: &Huffman, ac: &Huffman) {
        let magnitude = |value: i32| -> (u32, u32) {
            let bits = 32 - value.unsigned_abs().leading_zeros();
            // Negative values are sent as the low bits of the value minus one.
            let low = if value < 0 { value - 1 } else { value };
            (bits, low as u32)
        };
        let difference = i32::from(block[0]) - i32::from(*previous_dc);
        *previous_dc = block[0];
        let (bits, low) = magnitude(difference);
        self.put(
            u32::from(dc.code[bits as usize]),
            u32::from(dc.size[bits as usize]),
        );
        self.put(low, bits);
        let mut run = 0;
        for &position in &NATURAL_ORDER[1..] {
            let value = i32::from(block[position]);
            if value == 0 {
                run += 1;
                continue;
            }
            while run > 15 {
                self.put(u32::from(ac.code[0xf0]), u32::from(ac.size[0xf0]));
                run -= 16;
            }
            let (bits, low) = magnitude(value);
            let symbol = (run << 4) + bits as usize;
            self.put(u32::from(ac.code[symbol]), u32::from(ac.size[symbol]));
            self.put(low, bits);
            run = 0;
        }
        if run > 0 {
            self.put(u32::from(ac.code[0]), u32::from(ac.size[0]));
        }
    }
}

fn encode_jpeg(image: &Rgb) -> Vec<u8> {
    let (width, height) = (image.width, image.height);
    let (mcus_x, mcus_y) = (width.div_ceil(16), height.div_ceil(16));
    // Blocks that hold picture. A luma block beyond them is written as a filler.
    let (luma_blocks_x, luma_blocks_y) = (width.div_ceil(8), height.div_ceil(8));

    // Colour conversion at full size, with libjpeg's fixed-point constants.
    let pixels = width * height;
    let (mut y, mut cb, mut cr) = (
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
        Vec::with_capacity(pixels),
    );
    for pixel in image.data.as_chunks::<3>().0 {
        let (r, g, b) = (
            i32::from(pixel[0]),
            i32::from(pixel[1]),
            i32::from(pixel[2]),
        );
        y.push(((19595 * r + 38470 * g + 7471 * b + 32768) >> 16) as u8);
        cb.push(((-11059 * r - 21709 * g + 32768 * b + (128 << 16) + 32767) >> 16) as u8);
        cr.push(((32768 * r - 27439 * g - 5329 * b + (128 << 16) + 32767) >> 16) as u8);
    }

    // Luma is padded by repeating its last column and row.
    let luma = {
        let (padded_width, padded_height) = (mcus_x * 16, mcus_y * 16);
        let mut samples = Vec::with_capacity(padded_width * padded_height);
        for row in 0..padded_height {
            let source = row.min(height - 1) * width;
            samples.extend_from_slice(&y[source..source + width]);
            samples.resize(samples.len() + padded_width - width, y[source + width - 1]);
        }
        Plane {
            width: padded_width,
            samples,
        }
    };
    // Chroma is halved both ways. libjpeg repeats the last column before averaging, repeats
    // the last row when the height is odd, and afterwards repeats the last averaged row.
    let halved = |plane: &[u8]| {
        let (padded_width, padded_height) = (mcus_x * 8, mcus_y * 8);
        let at = |row: usize, column: usize| {
            u32::from(plane[row.min(height - 1) * width + column.min(width - 1)])
        };
        let mut samples = Vec::with_capacity(padded_width * padded_height);
        for row in 0..height.div_ceil(2) {
            for column in 0..padded_width {
                // The rounding alternates 1, 2 along the row so that it does not drift.
                let bias = 1 + (column as u32 & 1);
                let sum = at(row * 2, column * 2)
                    + at(row * 2, column * 2 + 1)
                    + at(row * 2 + 1, column * 2)
                    + at(row * 2 + 1, column * 2 + 1)
                    + bias;
                samples.push((sum >> 2) as u8);
            }
        }
        let last = samples[samples.len() - padded_width..].to_vec();
        while samples.len() < padded_width * padded_height {
            samples.extend_from_slice(&last);
        }
        Plane {
            width: padded_width,
            samples,
        }
    };
    let (cb, cr) = (halved(&cb), halved(&cr));

    let tables = [quant_table(&LUMA_QUANT), quant_table(&CHROMA_QUANT)];
    let divisors = tables.map(|table| table.map(divisor));
    let huffman = [&DC_LUMA, &AC_LUMA, &DC_CHROMA, &AC_CHROMA];

    let mut out = vec![0xff, 0xd8];
    let segment = |out: &mut Vec<u8>, marker: u8, body: &[u8]| {
        out.extend_from_slice(&[0xff, marker]);
        out.extend_from_slice(&(body.len() as u16 + 2).to_be_bytes());
        out.extend_from_slice(body);
    };
    // JFIF 1.01, no density.
    segment(&mut out, 0xe0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    for (index, table) in tables.iter().enumerate() {
        let mut body = vec![index as u8];
        body.extend(NATURAL_ORDER.iter().map(|&position| table[position] as u8));
        segment(&mut out, 0xdb, &body);
    }
    let mut frame = vec![8];
    frame.extend_from_slice(&(height as u16).to_be_bytes());
    frame.extend_from_slice(&(width as u16).to_be_bytes());
    frame.extend_from_slice(&[3, 1, 0x22, 0, 2, 0x11, 1, 3, 0x11, 1]);
    segment(&mut out, 0xc0, &frame);
    for (index, table) in huffman.iter().enumerate() {
        // Table class in the high half: 0 for DC, 1 for AC. Table number in the low half.
        let mut body = vec![((index as u8 & 1) << 4) | (index as u8 >> 1)];
        body.extend_from_slice(&table.0);
        body.extend_from_slice(table.1);
        segment(&mut out, 0xc4, &body);
    }
    segment(&mut out, 0xda, &[3, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0]);

    let codes = huffman.map(Huffman::new);
    let mut bits = Bits {
        out,
        ..Bits::default()
    };
    let mut previous_dc = [0i16; 3];
    let filler = |dc: i16| {
        let mut block = [0i16; 64];
        block[0] = dc;
        block
    };
    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            // Four luma blocks. One past the picture's edge repeats the DC value of the block
            // written before it and has nothing else.
            let mut last_dc = 0;
            for index in 0..4 {
                let (x, y) = (mcu_x * 2 + index % 2, mcu_y * 2 + index / 2);
                let block = if x < luma_blocks_x && y < luma_blocks_y {
                    luma.block(x, y, &divisors[0])
                } else {
                    filler(last_dc)
                };
                last_dc = block[0];
                bits.block(&block, &mut previous_dc[0], &codes[0], &codes[1]);
            }
            for (component, plane) in [(1, &cb), (2, &cr)] {
                let block = plane.block(mcu_x, mcu_y, &divisors[1]);
                bits.block(&block, &mut previous_dc[component], &codes[2], &codes[3]);
            }
        }
    }
    // The last byte is filled with ones.
    bits.put(0x7f, 7);
    let mut out = bits.out;
    out.extend_from_slice(&[0xff, 0xd9]);
    out
}

#[cfg(test)]
pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same picture the reference script draws: smooth in places, noisy in others.
    pub(crate) fn pattern(width: usize, height: usize) -> Rgb {
        let mut data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                data.push(((x * 5 + y * 3) % 256) as u8);
                data.push(((x * x + y * 7) % 256) as u8);
                data.push((((x ^ y) * 13 + x * y) % 256) as u8);
            }
        }
        Rgb {
            width,
            height,
            data,
        }
    }

    // Lengths and FNV-1a hashes of what Pillow 13.0.0 produces for the same pictures.
    #[test]
    fn resize_matches_pillow() {
        for ((width, height), (to_width, to_height), hash) in [
            ((37, 29), (61, 50), 0x85dc0975b69c30e2u64),
            // The height shrinks far more than the width, so the columns are resized first.
            ((90, 120), (60, 40), 0x2804d328e17fb9ff),
            ((64, 64), (64, 100), 0x65e57d38a699d18b),
            ((193, 195), (382, 386), 0x96faa6b4700c00e2),
        ] {
            let resized = pattern(width, height).resize_bilinear(to_width, to_height);
            assert_eq!((resized.width, resized.height), (to_width, to_height));
            assert_eq!(
                fnv1a(&resized.data),
                hash,
                "{width} × {height} to {to_width} × {to_height}"
            );
        }
    }

    #[test]
    fn jpeg_matches_pillow() {
        // Sizes that are and are not whole blocks, odd sizes, and a single pixel.
        for ((width, height), length, hash) in [
            ((33, 27), 1411, 0xeadcda9ba97de2d4u64),
            ((48, 32), 1816, 0xb5facf4dd5c93a1d),
            ((17, 40), 1224, 0x781b45dc34d33f8f),
            ((1, 1), 631, 0x3123b02d6e398248),
            ((386, 389), 119063, 0x0e17713ca5e94508),
        ] {
            let jpeg = pattern(width, height).jpeg_quality_90();
            assert_eq!(jpeg.len(), length, "{width} × {height}");
            assert_eq!(fnv1a(&jpeg), hash, "{width} × {height}");
        }
    }

    #[test]
    fn divisors_round_like_division() {
        for quant in 1..=255u16 {
            let d = divisor(quant);
            let by = u32::from(quant) * 8;
            for value in 0..=16384u32 {
                assert_eq!(
                    ((value + d.correction) * d.reciprocal) >> d.shift,
                    (value + by / 2) / by,
                    "quant {quant}, value {value}"
                );
            }
        }
    }
}
