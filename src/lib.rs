/// Basic, general types
pub mod types {

    /*** Basic linear algebra types ***/
    pub type Scalar = f32;

    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    pub struct Vector2(pub Scalar, pub Scalar);
    #[derive(Default, Clone, Copy, Debug, PartialEq)]
    pub struct Vector3(pub Scalar, pub Scalar, pub Scalar);
    pub struct Basis3(pub Vector3, pub Vector3, pub Vector3);

    impl std::ops::Add<Vector2> for Vector2 {
        type Output = Vector2;
        fn add(self, rhs: Vector2) -> Vector2 {
            Vector2(self.0 + rhs.0, self.1 + rhs.1)
        }
    }

    impl std::ops::Add<Vector3> for Vector3 {
        type Output = Vector3;
        fn add(self, rhs: Vector3) -> Vector3 {
            Vector3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
        }
    }

    impl<'a, 'b> std::ops::Add<&'b Vector2> for &'a Vector2 {
        type Output = Vector2;
        fn add(self, rhs: &'b Vector2) -> Vector2 {
            Vector2(self.0 + rhs.0, self.1 + rhs.1)
        }
    }

    impl<'a, 'b> std::ops::Add<&'b Vector3> for &'a Vector3 {
        type Output = Vector3;
        fn add(self, rhs: &'b Vector3) -> Vector3 {
            Vector3(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
        }
    }

    impl std::ops::Sub<Vector2> for Vector2 {
        type Output = Vector2;
        fn sub(self, rhs: Vector2) -> Vector2 {
            Vector2(self.0 - rhs.0, self.1 - rhs.1)
        }
    }

    impl std::ops::Sub<Vector3> for Vector3 {
        type Output = Vector3;
        fn sub(self, rhs: Vector3) -> Vector3 {
            Vector3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
        }
    }

    impl<'a, 'b> std::ops::Sub<&'b Vector2> for &'a Vector2 {
        type Output = Vector2;
        fn sub(self, rhs: &'b Vector2) -> Vector2 {
            Vector2(self.0 - rhs.0, self.1 - rhs.1)
        }
    }

    impl<'a, 'b> std::ops::Sub<&'b Vector3> for &'a Vector3 {
        type Output = Vector3;
        fn sub(self, rhs: &'b Vector3) -> Vector3 {
            Vector3(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
        }
    }

    impl std::ops::Mul<Scalar> for Vector2 {
        type Output = Vector2;
        fn mul(self, rhs: Scalar) -> Vector2 {
            Vector2(self.0 * rhs, self.1 * rhs)
        }
    }

    impl std::ops::Mul<Scalar> for Vector3 {
        type Output = Vector3;
        fn mul(self, rhs: Scalar) -> Vector3 {
            Vector3(self.0 * rhs, self.1 * rhs, self.2 * rhs)
        }
    }

    impl Vector2 {
        pub fn len(&self) -> Scalar {
            (Scalar::powi(self.0, 2) + Scalar::powi(self.1, 2)).sqrt()
        }

        pub fn normalized(self) -> Vector2 {
            self * (1.0 / self.len())
        }

        pub fn dot(&self, rhs: &Vector2) -> Scalar {
            (self.0 * rhs.0) + (self.1 * rhs.1)
        }

        // cross product
    }

    impl Vector3 {
        pub fn len(&self) -> Scalar {
            (Scalar::powi(self.0, 2) + Scalar::powi(self.1, 2) + Scalar::powi(self.2, 2)).sqrt()
        }

        pub fn normalized(self) -> Vector3 {
            self * (1.0 / self.len())
        }

        pub fn dot(&self, rhs: &Vector3) -> Scalar {
            (self.0 * rhs.0) + (self.1 * rhs.1) + (self.2 * rhs.2)
        }

        // cross product
    }

    /// Arbitrarily-sized 2D array of T values
    pub struct Matrix<T> {
        /// Flat array to store matrix in column-major order
        array: Vec<T>,
        rows: usize,
        cols: usize,
    }

    impl<T> Matrix<T> {
        pub fn new(rows: usize, cols: usize) -> Self
        where
            T: Default + Copy,
        {
            Self {
                array: [T::default()].repeat(rows * cols),
                rows,
                cols,
            }
        }

        pub fn get(&self, idx: (usize, usize)) -> &T {
            &self.array[idx.0 + idx.1 * self.rows]
        }

        pub fn rows(&self) -> usize {
            self.rows
        }

        pub fn cols(&self) -> usize {
            self.cols
        }

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

        // determinant

        // product
    }

    impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
        type Output = T;

        fn index(&self, idx: [usize; 2]) -> &T {
            &self.array[idx[0] + idx[1] * self.rows]
        }
    }

    impl<T> std::ops::IndexMut<[usize; 2]> for Matrix<T> {
        fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
            &mut self.array[idx[0] + idx[1] * self.rows]
        }
    }

    impl std::convert::From<Vector2> for Matrix<Scalar> {
        /// Returns a 2D column vector matrix
        fn from(vec: Vector2) -> Self {
            let mut matrix = Self::new(2, 1);
            matrix[[0, 0]] = vec.0;
            matrix[[1, 0]] = vec.1;
            matrix
        }
    }

    impl std::convert::From<Vector3> for Matrix<Scalar> {
        /// Returns a 2D column vector matrix
        fn from(vec: Vector3) -> Self {
            let mut matrix = Self::new(3, 1);
            matrix[[0, 0]] = vec.0;
            matrix[[1, 0]] = vec.1;
            matrix[[2, 0]] = vec.1;
            matrix
        }
    }

    impl std::convert::From<Matrix<Scalar>> for Vector2 {
        /// Takes a 2D column vector matrix
        fn from(matrix: Matrix<Scalar>) -> Self {
            Vector2(matrix[[0, 0]], matrix[[1, 0]])
        }
    }

    impl std::convert::From<Matrix<Scalar>> for Vector3 {
        /// Takes a 3D column vector matrix
        fn from(matrix: Matrix<Scalar>) -> Self {
            Vector3(matrix[[0, 0]], matrix[[1, 0]], matrix[[2, 0]])
        }
    }

    /// Parametric equations. Returns a point as a vector.
    pub type ParametricEq1 = fn(Scalar) -> Vector3;
    pub type ParametricEq2 = fn(Scalar, Scalar) -> Vector3;

    /// Implicit equation. Returns whether point satisfies equation.
    pub type ImplicitEq<T> = fn(T) -> bool;

    /*** Basic graphics types ***/
    /// Color channel in range [0,1]
    pub type ColorChannel = f32;

    /// Saturates value to within range [0,1]
    pub fn saturate_to_range(color: ColorChannel) -> ColorChannel {
        if color < 0.0 {
            return 0.0;
        } else if color > 1.0 {
            return 1.0;
        } else {
            return color;
        }
    }

    /// Converts one RGB8 channel to ColorChannel
    pub fn rgb8_to_colorchannel(color: u8) -> ColorChannel {
        saturate_to_range((color as ColorChannel) / 255.0)
    }

    /// RGB color
    #[derive(Default, Copy, Clone, Debug, PartialEq)]
    pub struct Color(pub ColorChannel, pub ColorChannel, pub ColorChannel);

    impl std::ops::Add<Color> for Color {
        type Output = Color;
        fn add(self, rhs: Color) -> Color {
            Color(
                saturate_to_range(self.0 + rhs.0),
                saturate_to_range(self.1 + rhs.1),
                saturate_to_range(self.2 + rhs.2),
            )
        }
    }

    impl std::ops::AddAssign for Color {
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
        fn mul(self, rhs: Scalar) -> Color {
            Color(
                saturate_to_range(self.0 * rhs),
                saturate_to_range(self.1 * rhs),
                saturate_to_range(self.2 * rhs),
            )
        }
    }

    impl std::ops::Mul<Color> for Color {
        type Output = Color;
        fn mul(self, rhs: Color) -> Color {
            Color(
                saturate_to_range(self.0 * rhs.0),
                saturate_to_range(self.1 * rhs.1),
                saturate_to_range(self.2 * rhs.2),
                //saturate_to_range(self.0 * rhs.0 * 10.0),
                //saturate_to_range(self.1 * rhs.1 * 10.0),
                //saturate_to_range(self.2 * rhs.2 * 10.0),
            )
        }
    }

    #[derive(Default, Copy, Clone, Debug, PartialEq)]
    pub struct Pixel(pub Color);

    /// 2D matrix of pixels
    pub type Image = Matrix<Pixel>;
}

/// Primitives for constructing a 3D scene
pub mod primitives {

    use crate::render::*;
    use crate::types::*;

    /// 3D scene of renderable objects
    pub struct Scene {
        pub background_color: Color,
        pub surfaces: Vec<Box<dyn Surface>>,
        /// Light sources themselves are not visible
        pub light_sources: Vec<Light>,
        /// RGB intensity
        pub ambient_light_intensity: Color,
    }

    impl Render for Scene {
        /// Checks scene for intersection
        fn hit(
            &self,
            ray: Ray,
            search_interval: (Scalar, Scalar),
            hit_rec: &mut HitRecord,
        ) -> bool {
            let mut hit = false;
            for surface in &self.surfaces {
                if surface.hit(ray, search_interval, hit_rec) {
                    hit = true;
                    break;
                }
            }
            hit
        }
    }

    pub struct Light {
        pub position: Vector3,
        /// RGB intensity
        pub intensity: Color,
    }

    /// Trait for defining entity structure
    pub trait Surface: Render {
        /// Returns (unit) normal vector of surface at point
        fn normal(&self, point: Vector3) -> Vector3;
        /// Returns material properties of surface
        fn material(&self) -> Material;
    }

    pub struct Sphere {
        pub center: Vector3,
        pub radius: Scalar,
        pub material: Material,
    }

    impl Surface for Sphere {
        /// Gives surface normal at a point.
        /// Assumes point is on surface.
        fn normal(&self, point: Vector3) -> Vector3 {
            (point - self.center) * (1.0 / self.radius)
        }

        fn material(&self) -> Material {
            self.material
        }
    }

    impl Render for Sphere {
        fn hit(
            &self,
            ray: Ray,
            search_interval: (Scalar, Scalar),
            hit_rec: &mut HitRecord,
        ) -> bool {
            let eye_to_center = ray.position - self.center;
            let mut discriminant = Scalar::powi(ray.direction.dot(&eye_to_center), 2);
            //println!("discriminant initial = {}", discriminant);
            discriminant -= ray.direction.dot(&ray.direction)
                * (eye_to_center.dot(&eye_to_center) - (self.radius * self.radius));
            if discriminant < 0.0 {
                return false;
            }
            /* Could change "- discriminant" to "+ discriminant"
            to get where the ray exits the sphere */
            let mut t: Scalar = -1.0 * ray.direction.dot(&eye_to_center) - discriminant.sqrt();
            t /= ray.direction.dot(&ray.direction);
            //hit_rec.surface = &self;
            hit_rec.t = t;
            let p_intersect = ray.position + ray.direction * t;
            hit_rec.normal = self.normal(p_intersect);
            hit_rec.material = self.material();
            t >= search_interval.0 && t < search_interval.1
        }
    }

    // pub struct Polygon

    // impl Surface for Polygon

    // impl Render for Polygon
}

/// Tools for rendering objects
pub mod render {

    use crate::primitives::*;
    use crate::types::*;

    /// Small positive bias to avoid imprecision errors
    pub const IMPRECISION_BIAS: Scalar = 0.000001;
    /// Search window limits
    pub const MIN_SEARCH: Scalar = 0.0 + IMPRECISION_BIAS;
    pub const MAX_SEARCH: Scalar = Scalar::INFINITY;

    /// Parametric ray
    #[derive(Default, Copy, Clone)]
    pub struct Ray {
        pub position: Vector3,
        pub direction: Vector3,
    }

    pub struct Camera {
        pub position: Vector3,
        /// Orthonormal right-handed camera space basis {u, v, w} where -w is the view direction
        pub basis: Basis3,
        /// Pixel resolution as (horizontal, vertical)
        /// Note: Both dimensions assumed to be even.
        pub resolution: (usize, usize),
        /// Euclidean distance from camera origin to image plane
        pub focal_length: Scalar,
    }

    impl Camera {
        // pub fn new(position: Vector3, view_direction: Vector3) -> Camera

        pub fn view_direction(&self) -> Vector3 {
            self.basis.2 * -1.0
        }

        /// Takes pixel coordinates and returns coordinates in {u, v} basis of camera space
        /// The (0, 0) pixel is in the top-left corner.
        pub fn pixel_to_camera_space(&self, pixel_idx: (usize, usize)) -> Vector2 {
            let x_size = self.resolution.0 as Scalar;
            let y_size = self.resolution.1 as Scalar;
            let (i, j) = pixel_idx;
            let l = x_size / -2.0;
            let t = y_size / 2.0;
            let u = l + (-2.0 * l) * (i as Scalar + 0.5) / x_size;
            let v = t + (-2.0 * t) * (j as Scalar + 0.5) / y_size;
            Vector2(u, v)
        }

        /// Returns view ray for given pixel coordinates.
        pub fn pixel_to_ray(&self, pixel_idx: (usize, usize)) -> Ray {
            let Vector2(u, v) = self.pixel_to_camera_space((pixel_idx.0, pixel_idx.1));
            let view_vec = self.view_direction();
            let u_vec = self.basis.0;
            let v_vec = self.basis.1;
            let e = self.position;
            let d = (view_vec * self.focal_length) + (u_vec * u) + (v_vec * v);
            Ray {
                position: e,
                direction: d,
            }
        }
    }

    /// Struct modeling apparent properties of a surface
    #[derive(Default, Copy, Clone)]
    pub struct Material {
        /// 10 = matte, 100 = mildly shiny, 1000 = really glossy, 10000 = mirror
        pub shininess: Scalar,
        /// Surface color
        pub diffuse_color: Color,
        pub specular_color: Color,
        pub ambient_color: Color,
        pub mirror_color: Color,
    }

    /// Stores rendering data about ray-surface intersection
    /// Note: Generally only valid when a surface is hit
    #[derive(Default)]
    pub struct HitRecord {
        /// The surface hit
        //pub surface: &dyn Surface,
        /// Ray t value of intersection
        pub t: Scalar,
        /// Surface normal at intersection point
        pub normal: Vector3,
        /// Material for the surface intersected
        pub material: Material,
    }

    /// Trait for defining entity rendering behavior
    pub trait Render {
        /// Returns the t value of intersection or None if no intersection
        /// Accepts ray, search interval as [t_near, t_far), and a hit record
        fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord)
            -> bool;
    }

    /// Gets the color seen by a ray in a scene.
    fn raycolor(scene: &Scene, ray: Ray, search_interval: (Scalar, Scalar), depth: usize) -> Color {
        //println!("Getting ray color...");
        pub const MAX_RECURSIVE_DEPTH: usize = 5;
        if depth > MAX_RECURSIVE_DEPTH {
            // Give up reflecting ray off mirroring surfaces
            return scene.background_color;
            //return Color(0.0, 0.0, 0.0);
        }

        let mut hit_rec = HitRecord::default();
        let mut shadow_rec = HitRecord::default();
        if scene.hit(ray, search_interval, &mut hit_rec) {
            // Ray hit a surface (with a given material) at point p_intersect
            let p_intersect = ray.position + (ray.direction * hit_rec.t);
            /*println!(
                "Ray hit a surface at ({},{},{}) with n = ({},{},{})! Shading...",
                p_intersect.0,
                p_intersect.1,
                p_intersect.2,
                hit_rec.normal.0,
                hit_rec.normal.1,
                hit_rec.normal.2
            );*/
            let material = hit_rec.material;

            // Add base color due to ambient lighting
            let mut color = material.ambient_color * scene.ambient_light_intensity;
            /*println!(
                "Color initial (just ambient lighting) = ({},{},{})",
                color.0, color.1, color.2
            );*/
            // Add color contribution for each light
            for light in &scene.light_sources {
                //println!("Adding light...");
                // Direction from p_intersect to light
                let light_direction = light.position - p_intersect;
                /*println!("Light direction = ({},{},{})",
                    light_direction.0, light_direction.1, light_direction.2);
                println!("Normal direction = ({},{},{})",
                    hit_rec.normal.0, hit_rec.normal.1, hit_rec.normal.2);*/
                let shadow_ray = Ray {
                    position: p_intersect,
                    direction: light_direction,
                };
                // Check if point is in shadow (i.e., shadow_ray hits something), shade if not
                if !scene.hit(shadow_ray, search_interval, &mut shadow_rec) {
                    //println!("Point is NOT in shadow relative to this light, shading...");
                    // Half vector bisecting angle between ray and shadow ray
                    let half_vec = (light_direction.normalized()
                        + ray.direction.normalized() * -1.0)
                        .normalized();
                    // Add Lambertian (color) contribution
                    color += material.diffuse_color
                        * light.intensity
                        * hit_rec.normal.dot(&light_direction).max(0.0);
                    /*println!(
                        "Color after Lambertian shading = ({},{},{})",
                        color.0, color.1, color.2
                    );*/
                    // Add Blinn-Phong (specular + shininess) contribution
                    color += material.specular_color
                        * light.intensity
                        * Scalar::powf(hit_rec.normal.dot(&half_vec), material.shininess);
                    /*println!(
                        "Color after Blinn-Phong shading = ({},{},{})",
                        color.0, color.1, color.2
                    );*/
                } else {
                    //println!("Point IS in shadow relative to this light, not shading");
                }
            }

            // Get mirror reflection contribution (if mirror color isn't black)
            if material.mirror_color != Color(0.0, 0.0, 0.0) {
                // Ray formed by reflecting off surface
                let reflect_ray = Ray {
                    position: p_intersect,
                    direction: ray.direction
                        - hit_rec.normal * hit_rec.normal.dot(&ray.direction) * 2.0,
                };
                //println!("Hit a mirror, reflecting...");
                // Recursive call to mimic reflection off chain of mirrors
                color += material.mirror_color
                    * raycolor(scene, reflect_ray, search_interval, depth + 1);
            }
            /*println!(
                "Color after mirror shading = ({},{},{})",
                color.0, color.1, color.2
            );*/
            return color;
        } else {
            //println!("Hit nothing, setting to background color...");
            return scene.background_color;
        }
    }

    pub fn render(camera: &Camera, scene: &Scene) -> Image {
        let x_size = camera.resolution.0;
        let y_size = camera.resolution.1;
        let mut frame = Image::new(x_size, y_size);
        for i in 0..x_size {
            for j in 0..y_size {
                //println!("Shading pixel ({},{})", i, j);
                let ray = camera.pixel_to_ray((i, j));
                // Search whole world (minmax rendering distance)
                let search_interval = (MIN_SEARCH, MAX_SEARCH);
                frame[[i, j]] = Pixel(raycolor(scene, ray, search_interval, 0));
            }
        }
        frame
    }
}

#[cfg(test)]
mod tests {
    use crate::primitives::*;
    use crate::render::*;
    use crate::types::*;

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
    #[test]
    fn camera_space_conversion_fits_corners() {
        let r: Scalar = 2.0;
        let l: Scalar = -r;
        let t: Scalar = 2.0;
        let b: Scalar = -t;
        let x_size = (r - l) as usize;
        let y_size = (t - b) as usize;
        let basis = Basis3(
            Vector3(0.0, 1.0, 0.0),
            Vector3(0.0, 0.0, 1.0),
            Vector3(1.0, 0.0, 0.0),
        );
        let camera = Camera {
            position: Vector3(16.0, 0.0, 0.0),
            basis,
            resolution: (x_size, y_size),
            focal_length: 8.0,
        };
        // Top-left
        let Vector2(mut u, mut v) = camera.pixel_to_camera_space((0, 0));
        assert_eq!(u, l + 0.5);
        assert_eq!(v, t - 0.5);
        // Top-right
        let Vector2(mut u, mut v) = camera.pixel_to_camera_space((x_size - 1, 0));
        assert_eq!(u, r - 0.5);
        assert_eq!(v, t - 0.5);
        // Bottom-left
        let Vector2(mut u, mut v) = camera.pixel_to_camera_space((0, y_size - 1));
        assert_eq!(u, l + 0.5);
        assert_eq!(v, b + 0.5);
        // Bottom-right
        let Vector2(mut u, mut v) = camera.pixel_to_camera_space((x_size - 1, y_size - 1));
        assert_eq!(u, r - 0.5);
        assert_eq!(v, b + 0.5);
    }
    #[test]
    fn ray_sphere_intersection() {
        let green = Color(
            rgb8_to_colorchannel(64),
            rgb8_to_colorchannel(255),
            rgb8_to_colorchannel(64),
        );
        let material = Material {
            shininess: 100.0,
            diffuse_color: green,
            specular_color: green,
            ambient_color: green,
            mirror_color: green,
        };
        let sphere = Sphere {
            center: Vector3(0.0, 0.0, 0.0),
            radius: 4.0,
            material,
        };

        let mut hit_rec = HitRecord::default();
        let search_interval = (MIN_SEARCH, MAX_SEARCH);

        let position = Vector3(16.0, 0.0, 0.0);
        let ray = Ray {
            position,
            direction: Vector3(0.0, 0.0, 0.0) - position,
        };

        assert!(sphere.hit(ray, search_interval, &mut hit_rec));
    }
}
