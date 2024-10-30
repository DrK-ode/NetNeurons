use crate::nnetwork::FloatType;

/// Specifies whether a pixel is red, green and/or blue.
pub type ColorKey = &'static dyn Fn((FloatType, FloatType)) -> [bool; 3];

/// Example [ColorKey] representing three overlapping spheres.
pub const RGB_VENN_DIAGRAM: &dyn Fn((FloatType, FloatType)) -> [bool; 3] =
    &|(x, y): (FloatType, FloatType)| {
        [
            (x - 0.2165).powi(2) + (y + 0.125).powi(2) < 0.25,
            (x + 0.2165).powi(2) + (y + 0.125).powi(2) < 0.25,
            x.powi(2) + (y - 0.25).powi(2) < 0.25,
        ]
    };
