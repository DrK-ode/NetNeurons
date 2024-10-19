use crate::nnetwork::FloatType;

pub enum Color {
    None = 0,
    Red = 1,
    Blue = 2,
    Both = 3,
}

impl From<Color> for (bool,bool)
{
    fn from(value: Color) -> Self {
        match value {
            Color::None => (false,false),
            Color::Red => (true,false),
            Color::Blue => (false,true),
            Color::Both => (true,true),
        }
    }
}

impl From<(bool,bool)> for Color{
    fn from(value: (bool,bool)) -> Self {
        let (is_red, is_blue) = value;
        if is_red {
            if is_blue{
                Color::Both
            }
            else{
                Color::Red
            }
        }
        else if is_blue{
            Color::Blue
        }
        else {
            Color::None
        }
    }
}

impl From<usize> for Color{
    fn from(n: usize) -> Self {
        match n {
            0 => Color::None,
            1 => Color::Red,
            2 => Color::Blue,
            3 => Color::Both,
            _ => panic!("Cannot create a Color from {n}")
        }
    }
}

impl From<Color> for usize {
    fn from(value: Color) -> Self {
        match value {
            Color::None => 0,
            Color::Red => 1,
            Color::Blue => 2,
            Color::Both => 3,
        }
    }
}

pub type ColorFunction = Box<dyn Fn((FloatType,FloatType))->(bool,bool)>;

pub struct ColorKey
{
    _function: ColorFunction,
}

impl ColorKey
{
    pub fn new(is_red_or_blue: ColorFunction) -> ColorKey {
        ColorKey {
            _function: is_red_or_blue,
        }
    }
    
    pub fn color(&self, coords: (FloatType,FloatType)) -> Color{
        (self._function)(coords).into()
    }
}
