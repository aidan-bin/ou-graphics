pub mod linalg {
    pub type Scalar = f32;

    /// 2D vector with x and y components
    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    pub struct Vector2(pub Scalar, pub Scalar);

    /// 3D vector with x, y, and z components
    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    pub struct Vector3(pub Scalar, pub Scalar, pub Scalar);

    /// Orthonormal basis for 3D space represented as three vectors (u, v, w)
    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    pub struct Basis3(pub Vector3, pub Vector3, pub Vector3);

    impl std::ops::Add for Vector2 {
        type Output = Self;
        #[inline]
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0, self.1 + rhs.1)
        }
    }

    impl std::ops::Add for Vector3 {
        type Output = Self;
        #[inline]
        fn add(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
        }
    }

    impl std::ops::Add<&Vector2> for &Vector2 {
        type Output = Vector2;
        #[inline]
        fn add(self, rhs: &Vector2) -> Vector2 {
            Vector2(self.0 + rhs.0, self.1 + rhs.1)
        }
    }

    impl std::ops::Add<&Vector3> for &Vector3 {
        type Output = Vector3;
        #[inline]
        fn add(self, rhs: &Vector3) -> Vector3 {
            Vector3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
        }
    }

    impl std::ops::Sub for Vector2 {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0, self.1 - rhs.1)
        }
    }

    impl std::ops::Sub for Vector3 {
        type Output = Self;
        #[inline]
        fn sub(self, rhs: Self) -> Self {
            Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
        }
    }

    impl std::ops::Sub<&Vector2> for &Vector2 {
        type Output = Vector2;
        #[inline]
        fn sub(self, rhs: &Vector2) -> Vector2 {
            Vector2(self.0 - rhs.0, self.1 - rhs.1)
        }
    }

    impl std::ops::Sub<&Vector3> for &Vector3 {
        type Output = Vector3;
        #[inline]
        fn sub(self, rhs: &Vector3) -> Vector3 {
            Vector3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
        }
    }

    impl std::ops::Mul<Scalar> for Vector2 {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: Scalar) -> Self {
            Self(self.0 * rhs, self.1 * rhs)
        }
    }

    impl std::ops::Mul<Scalar> for Vector3 {
        type Output = Self;
        #[inline]
        fn mul(self, rhs: Scalar) -> Self {
            Self(self.0 * rhs, self.1 * rhs, self.2 * rhs)
        }
    }

    impl std::ops::Mul<Vector2> for Scalar {
        type Output = Vector2;
        #[inline]
        fn mul(self, rhs: Vector2) -> Vector2 {
            Vector2(self * rhs.0, self * rhs.1)
        }
    }

    impl std::ops::Mul<Vector3> for Scalar {
        type Output = Vector3;
        #[inline]
        fn mul(self, rhs: Vector3) -> Vector3 {
            Vector3(self * rhs.0, self * rhs.1, self * rhs.2)
        }
    }

    // Add division by scalar
    impl std::ops::Div<Scalar> for Vector2 {
        type Output = Self;
        #[inline]
        fn div(self, rhs: Scalar) -> Self {
            Self(self.0 / rhs, self.1 / rhs)
        }
    }

    impl std::ops::Div<Scalar> for Vector3 {
        type Output = Self;
        #[inline]
        fn div(self, rhs: Scalar) -> Self {
            Self(self.0 / rhs, self.1 / rhs, self.2 / rhs)
        }
    }

    impl std::ops::Neg for Vector2 {
        type Output = Self;
        #[inline]
        fn neg(self) -> Self {
            Self(-self.0, -self.1)
        }
    }

    impl std::ops::Neg for Vector3 {
        type Output = Self;
        #[inline]
        fn neg(self) -> Self {
            Self(-self.0, -self.1, -self.2)
        }
    }

    impl Vector2 {
        pub const ZERO: Vector2 = Vector2(0.0, 0.0);
        pub const ONE: Vector2 = Vector2(1.0, 1.0);
        pub const X: Vector2 = Vector2(1.0, 0.0);
        pub const Y: Vector2 = Vector2(0.0, 1.0);

        #[inline]
        pub fn len(&self) -> Scalar {
            (self.0.powi(2) + self.1.powi(2)).sqrt()
        }

        #[inline]
        #[must_use]
        pub fn normalized(self) -> Self {
            let len = self.len();
            if len <= f32::EPSILON {
                self
            } else {
                self * (1.0 / len)
            }
        }

        #[inline]
        pub fn dot(&self, rhs: &Self) -> Scalar {
            self.0 * rhs.0 + self.1 * rhs.1
        }

        #[inline]
        pub fn cross(&self, rhs: &Self) -> Scalar {
            self.0 * rhs.1 - self.1 * rhs.0
        }

        #[inline]
        pub fn len_squared(&self) -> Scalar {
            self.0.powi(2) + self.1.powi(2)
        }

        #[inline]
        pub fn distance(&self, other: &Self) -> Scalar {
            (*self - *other).len()
        }

        #[inline]
        pub fn distance_squared(&self, other: &Self) -> Scalar {
            (*self - *other).len_squared()
        }
    }

    impl Vector3 {
        pub const ZERO: Vector3 = Vector3(0.0, 0.0, 0.0);
        pub const ONE: Vector3 = Vector3(1.0, 1.0, 1.0);
        pub const X: Vector3 = Vector3(1.0, 0.0, 0.0);
        pub const Y: Vector3 = Vector3(0.0, 1.0, 0.0);
        pub const Z: Vector3 = Vector3(0.0, 0.0, 1.0);
        pub const ORIGIN: Vector3 = Vector3(0.0, 0.0, 0.0);

        #[inline]
        pub fn len(&self) -> Scalar {
            (self.0.powi(2) + self.1.powi(2) + self.2.powi(2)).sqrt()
        }

        #[inline]
        #[must_use]
        pub fn normalized(self) -> Self {
            let len = self.len();
            if len <= f32::EPSILON {
                self
            } else {
                self * (1.0 / len)
            }
        }

        #[inline]
        pub fn dot(&self, rhs: &Self) -> Scalar {
            self.0 * rhs.0 + self.1 * rhs.1 + self.2 * rhs.2
        }

        #[inline]
        pub fn cross(&self, rhs: &Self) -> Self {
            Self(
                self.1 * rhs.2 - self.2 * rhs.1,
                self.2 * rhs.0 - self.0 * rhs.2,
                self.0 * rhs.1 - self.1 * rhs.0,
            )
        }

        #[inline]
        pub fn len_squared(&self) -> Scalar {
            self.0.powi(2) + self.1.powi(2) + self.2.powi(2)
        }

        #[inline]
        pub fn distance(&self, other: &Self) -> Scalar {
            (*self - *other).len()
        }

        #[inline]
        pub fn distance_squared(&self, other: &Self) -> Scalar {
            (*self - *other).len_squared()
        }
    }

    /// Arbitrarily-sized 2D array of T values
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Matrix<T> {
        /// Flat array to store matrix in column-major order
        array: Vec<T>,
        rows: usize,
        cols: usize,
    }

    impl<T: Default + Copy> Matrix<T> {
        #[inline]
        pub fn new(rows: usize, cols: usize) -> Self {
            Self {
                array: vec![T::default(); rows * cols],
                rows,
                cols,
            }
        }
    }

    impl<T> Matrix<T> {
        #[inline]
        pub fn get(&self, idx: (usize, usize)) -> &T {
            &self.array[idx.0 + idx.1 * self.rows]
        }

        #[inline]
        pub fn get_mut(&mut self, idx: (usize, usize)) -> &mut T {
            &mut self.array[idx.0 + idx.1 * self.rows]
        }

        #[inline]
        pub fn set(&mut self, idx: (usize, usize), value: T) {
            self.array[idx.0 + idx.1 * self.rows] = value;
        }

        #[inline]
        pub fn rows(&self) -> usize {
            self.rows
        }

        #[inline]
        pub fn cols(&self) -> usize {
            self.cols
        }

        #[inline]
        pub fn transpose(&self) -> Matrix<T>
        where
            T: Default + Copy,
        {
            let mut matrix = Self::new(self.cols, self.rows);
            for i in 0..self.rows {
                for j in 0..self.cols {
                    matrix[[j, i]] = self[[i, j]];
                }
            }
            matrix
        }

        /// Returns an iterator over the matrix elements in column-major order
        #[inline]
        pub fn iter(&self) -> std::slice::Iter<T> {
            self.array.iter()
        }

        /// Returns a mutable iterator over the matrix elements in column-major order
        #[inline]
        pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
            self.array.iter_mut()
        }
        // TODO: determinant
        // TODO: product
    }

    impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
        type Output = T;
        #[inline]
        fn index(&self, idx: [usize; 2]) -> &T {
            &self.array[idx[0] + idx[1] * self.rows]
        }
    }

    impl<T> std::ops::IndexMut<[usize; 2]> for Matrix<T> {
        #[inline]
        fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
            &mut self.array[idx[0] + idx[1] * self.rows]
        }
    }

    impl std::convert::From<Vector2> for Matrix<Scalar> {
        /// Returns a 2D column vector matrix
        #[inline]
        fn from(vec: Vector2) -> Self {
            let mut matrix = Self::new(2, 1);
            matrix[[0, 0]] = vec.0;
            matrix[[1, 0]] = vec.1;
            matrix
        }
    }

    impl std::convert::From<Vector3> for Matrix<Scalar> {
        /// Returns a 3D column vector matrix
        #[inline]
        fn from(vec: Vector3) -> Self {
            let mut matrix = Self::new(3, 1);
            matrix[[0, 0]] = vec.0;
            matrix[[1, 0]] = vec.1;
            matrix[[2, 0]] = vec.2;
            matrix
        }
    }

    impl std::convert::From<Matrix<Scalar>> for Vector2 {
        /// Takes a 2D column vector matrix
        #[inline]
        fn from(matrix: Matrix<Scalar>) -> Self {
            Vector2(matrix[[0, 0]], matrix[[1, 0]])
        }
    }

    impl std::convert::From<Matrix<Scalar>> for Vector3 {
        /// Takes a 3D column vector matrix
        #[inline]
        fn from(matrix: Matrix<Scalar>) -> Self {
            Vector3(matrix[[0, 0]], matrix[[1, 0]], matrix[[2, 0]])
        }
    }

    /// Parametric equations. Returns a point as a vector.
    pub type ParametricEq1 = fn(Scalar) -> Vector3;
    pub type ParametricEq2 = fn(Scalar, Scalar) -> Vector3;

    /// Implicit equation. Returns whether point satisfies equation.
    pub type ImplicitEq<T> = fn(T) -> bool;
}

pub mod image {
    use super::linalg::Matrix;
    use super::linalg::Scalar;
    /// Color channel in range [0,1]
    pub type ColorChannel = f32;

    /// Saturates value to within range [0,1]
    #[inline]
    pub fn saturate_to_range(color: ColorChannel) -> ColorChannel {
        color.clamp(0.0, 1.0)
    }

    /// Converts one RGB8 channel to ColorChannel
    #[inline]
    pub fn rgb8_to_colorchannel(color: u8) -> ColorChannel {
        saturate_to_range((color as ColorChannel) / 255.0)
    }

    /// RGB color
    #[derive(Default, Copy, Clone, Debug, PartialEq)]
    pub struct Color(pub ColorChannel, pub ColorChannel, pub ColorChannel);

    impl std::ops::Add<Color> for Color {
        type Output = Color;
        #[inline]
        fn add(self, rhs: Color) -> Color {
            Color(
                saturate_to_range(self.0 + rhs.0),
                saturate_to_range(self.1 + rhs.1),
                saturate_to_range(self.2 + rhs.2),
            )
        }
    }

    impl std::ops::AddAssign for Color {
        #[inline]
        fn add_assign(&mut self, rhs: Color) {
            *self = Color(
                saturate_to_range(self.0 + rhs.0),
                saturate_to_range(self.1 + rhs.1),
                saturate_to_range(self.2 + rhs.2),
            );
        }
    }

    impl std::ops::Mul<Scalar> for Color {
        type Output = Color;
        #[inline]
        fn mul(self, rhs: Scalar) -> Color {
            Color(
                saturate_to_range(self.0 * rhs),
                saturate_to_range(self.1 * rhs),
                saturate_to_range(self.2 * rhs),
            )
        }
    }

    impl std::ops::Mul<Color> for Scalar {
        type Output = Color;
        #[inline]
        fn mul(self, rhs: Color) -> Color {
            Color(
                saturate_to_range(self * rhs.0),
                saturate_to_range(self * rhs.1),
                saturate_to_range(self * rhs.2),
            )
        }
    }

    impl std::ops::Mul<Color> for Color {
        type Output = Color;
        #[inline]
        fn mul(self, rhs: Color) -> Color {
            Color(
                saturate_to_range(self.0 * rhs.0),
                saturate_to_range(self.1 * rhs.1),
                saturate_to_range(self.2 * rhs.2),
            )
        }
    }

    impl Color {
        pub const BLACK: Color = Color(0.0, 0.0, 0.0);
        pub const WHITE: Color = Color(1.0, 1.0, 1.0);
        pub const RED: Color = Color(1.0, 0.0, 0.0);
        pub const GREEN: Color = Color(0.0, 1.0, 0.0);
        pub const BLUE: Color = Color(0.0, 0.0, 1.0);
        pub const YELLOW: Color = Color(1.0, 1.0, 0.0);
        pub const CYAN: Color = Color(0.0, 1.0, 1.0);
        pub const MAGENTA: Color = Color(1.0, 0.0, 1.0);
        pub const ORANGE: Color = Color(1.0, 0.5, 0.0);
        pub const PURPLE: Color = Color(0.5, 0.0, 1.0);
        pub const PINK: Color = Color(1.0, 0.75, 0.8);
        pub const BROWN: Color = Color(0.6, 0.3, 0.1);
        pub const GRAY: Color = Color(0.5, 0.5, 0.5);
        pub const LIGHT_GRAY: Color = Color(0.75, 0.75, 0.75);
        pub const DARK_GRAY: Color = Color(0.25, 0.25, 0.25);

        /// Create a color from RGB values in the range [0, 255]
        #[inline]
        pub fn from_rgb8(r: u8, g: u8, b: u8) -> Self {
            Color(
                rgb8_to_colorchannel(r),
                rgb8_to_colorchannel(g),
                rgb8_to_colorchannel(b),
            )
        }

        /// Convert to RGB8 tuple
        #[inline]
        pub fn to_rgb8(self) -> (u8, u8, u8) {
            (
                (self.0 * 255.0) as u8,
                (self.1 * 255.0) as u8,
                (self.2 * 255.0) as u8,
            )
        }

        /// Linear interpolation between two colors
        #[inline]
        pub fn lerp(self, other: Color, t: ColorChannel) -> Color {
            self * (1.0 - t) + other * t
        }
    }

    #[derive(Default, Copy, Clone, Debug, PartialEq)]
    pub struct Pixel(pub Color);

    /// 2D matrix of pixels
    pub type Image = Matrix<Pixel>;
}

#[cfg(test)]
mod tests {
    use super::image::*;
    use super::linalg::*;

    /// Vector tests
    #[test]
    fn vector2_add() {
        let v1 = Vector2(1.0, 2.0);
        let v2 = Vector2(2.0, 3.0);
        assert_eq!(v1 + v2, Vector2(3.0, 5.0));
    }

    #[test]
    fn vector2_add_wrong() {
        let v1 = Vector2(1.0, 2.0);
        let v2 = Vector2(2.0, 3.0);
        assert_ne!(v1 + v2, Vector2(3.0, 8.0));
    }

    #[test]
    fn vector3_add() {
        let v1 = Vector3(1.0, 2.0, 3.0);
        let v2 = Vector3(2.0, 3.0, 4.0);
        assert_eq!(v1 + v2, Vector3(3.0, 5.0, 7.0));
    }

    #[test]
    fn vector2_sub() {
        let v1 = Vector2(1.0, 2.0);
        let v2 = Vector2(2.0, 3.0);
        assert_eq!(v1 - v2, Vector2(-1.0, -1.0));
    }

    #[test]
    fn vector3_sub() {
        let v1 = Vector3(1.0, 2.0, 3.0);
        let v2 = Vector3(2.0, 3.0, 4.0);
        assert_eq!(v1 - v2, Vector3(-1.0, -1.0, -1.0));
    }

    #[test]
    fn vector2_mul() {
        let v = Vector2(1.0, 2.0);
        assert_eq!(v * 4.0, Vector2(4.0, 8.0));
    }

    #[test]
    fn vector3_mul() {
        let v = Vector3(1.0, 2.0, 3.0);
        assert_eq!(v * 4.0, Vector3(4.0, 8.0, 12.0));
    }

    #[test]
    fn vector2_dot() {
        let v1 = Vector2(1.0, 2.0);
        let v2 = Vector2(2.0, 3.0);
        let expected = 8.0;
        assert_eq!(v1.dot(&v2), expected);
        assert_eq!(v2.dot(&v1), expected);
    }

    #[test]
    fn vector3_dot() {
        let v1 = Vector3(1.0, 2.0, 3.0);
        let v2 = Vector3(2.0, 3.0, 4.0);
        let expected = 20.0;
        assert_eq!(v1.dot(&v2), expected);
        assert_eq!(v2.dot(&v1), expected);
    }

    /// Image and color tests
    #[test]
    fn colorchannel_saturation() {
        assert_eq!(saturate_to_range(-0.5), 0.0);
        assert_eq!(saturate_to_range(0.5), 0.5);
        assert_eq!(saturate_to_range(1.5), 1.0);
    }

    #[test]
    fn rgb8_conversion() {
        assert_eq!(rgb8_to_colorchannel(0), 0.0);
        assert_eq!(rgb8_to_colorchannel(255), 1.0);
        assert_eq!(rgb8_to_colorchannel(51), 51.0 / 255.0);
    }

    #[test]
    fn color_scaling() {
        assert_eq!(Color(1.0, 1.0, 1.0) * 0.5, Color(0.5, 0.5, 0.5));
        assert_eq!(Color(0.5, 0.5, 0.5) * 0.5, Color(0.25, 0.25, 0.25));
        assert_eq!(Color(1.0, 1.0, 1.0) * 2.0, Color(1.0, 1.0, 1.0));
    }

    #[test]
    fn color_multiplication() {
        let red = Color(1.0, 0.0, 0.0);
        let green = Color(0.0, 1.0, 0.0);
        let white = Color(1.0, 1.0, 1.0);
        let gray = Color(0.5, 0.5, 0.5);

        assert_eq!(red * green, Color(0.0, 0.0, 0.0)); // No common components
        assert_eq!(white * gray, Color(0.5, 0.5, 0.5)); // Scale by gray
        assert_eq!(red * white, Color(1.0, 0.0, 0.0)); // Identity with white

        assert_eq!(red * green, green * red);
        assert_eq!(white * gray, gray * white);
    }

    #[test]
    fn image_new_and_get() {
        let x_size: usize = 3;
        let y_size: usize = 3;
        let mut image = Image::new(x_size, y_size);
        for i in 0..x_size {
            for j in 0..y_size {
                image[[i, j]] = Pixel(Color(
                    (i as ColorChannel) * (1.0 / 255.0) + 10.0,
                    (j as ColorChannel) * (1.0 / 255.0) + 10.0,
                    0.5,
                ));
            }
        }
        for i in 0..x_size {
            for j in 0..y_size {
                let pixel = image[[i, j]];
                assert_eq!(
                    pixel,
                    Pixel(Color(
                        (i as ColorChannel) * (1.0 / 255.0) + 10.0,
                        (j as ColorChannel) * (1.0 / 255.0) + 10.0,
                        0.5,
                    ))
                );
            }
        }
    }
}
