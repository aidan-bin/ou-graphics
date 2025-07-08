/// Primitives for constructing a 3D scene
use crate::render::*;
use crate::types::image::*;
use crate::types::linalg::*;

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
    /// Checks scene for intersection, returns closest hit
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool {
        let mut closest_hit = false;
        let mut closest_t = search_interval.1; // Start with far limit

        for surface in &self.surfaces {
            let mut temp_hit_rec = HitRecord::default();
            if surface.hit(ray, (search_interval.0, closest_t), &mut temp_hit_rec)
                && temp_hit_rec.t < closest_t
            {
                closest_t = temp_hit_rec.t;
                *hit_rec = temp_hit_rec; // Copy the closest hit
                closest_hit = true;
            }
        }
        closest_hit
    }
}

pub struct Light {
    pub position: Vector3,
    /// RGB intensity
    pub intensity: Color,
}

impl Light {
    /// Create a new point light
    pub fn new(position: Vector3, intensity: Color) -> Self {
        Self {
            position,
            intensity,
        }
    }

    /// Create a white light with given intensity
    pub fn white(position: Vector3, intensity: Scalar) -> Self {
        Self {
            position,
            intensity: Color::WHITE * intensity,
        }
    }

    /// Get the direction from a point to this light
    pub fn direction_from(&self, point: Vector3) -> Vector3 {
        (self.position - point).normalized()
    }

    /// Get the distance from a point to this light
    pub fn distance_from(&self, point: Vector3) -> Scalar {
        (self.position - point).len()
    }
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
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool {
        let eye_to_center = ray.position - self.center;

        // Quadratic equation coefficients: at^2 + bt + c = 0
        let a = ray.direction.dot(&ray.direction); // Should be 1.0 for normalized rays
        let b = 2.0 * eye_to_center.dot(&ray.direction);
        let c = eye_to_center.dot(&eye_to_center) - self.radius * self.radius;

        let discriminant = b * b - 4.0 * a * c;
        if discriminant < 0.0 {
            return false; // No intersection
        }

        let sqrt_discriminant = discriminant.sqrt();

        // Try the closer intersection first
        let mut t = (-b - sqrt_discriminant) / (2.0 * a);
        if t < search_interval.0 || t >= search_interval.1 {
            // Try the farther intersection
            t = (-b + sqrt_discriminant) / (2.0 * a);
            if t < search_interval.0 || t >= search_interval.1 {
                return false; // Both intersections outside interval
            }
        }

        // We have a valid intersection
        hit_rec.t = t;
        let p_intersect = ray.position + ray.direction * t;
        hit_rec.normal = self.normal(p_intersect);
        hit_rec.material = self.material();
        true
    }
}

impl Sphere {
    /// Create a new sphere
    pub fn new(center: Vector3, radius: Scalar, material: Material) -> Self {
        Self {
            center,
            radius,
            material,
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            background_color: Color::BLACK,
            surfaces: Vec::new(),
            light_sources: Vec::new(),
            ambient_light_intensity: Color(0.1, 0.1, 0.1), // Soft ambient light
        }
    }
}

impl Scene {
    /// Create a new empty scene with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new scene with custom settings
    pub fn with_settings(background_color: Color, ambient_light_intensity: Color) -> Self {
        Self {
            background_color,
            surfaces: Vec::new(),
            light_sources: Vec::new(),
            ambient_light_intensity,
        }
    }

    /// Add a surface to the scene
    pub fn add_surface(&mut self, surface: Box<dyn Surface>) {
        self.surfaces.push(surface);
    }

    /// Add a light to the scene
    pub fn add_light(&mut self, light: Light) {
        self.light_sources.push(light);
    }

    /// Set background color
    pub fn set_background(&mut self, color: Color) {
        self.background_color = color;
    }

    /// Set ambient light intensity
    pub fn set_ambient_light(&mut self, intensity: Color) {
        self.ambient_light_intensity = intensity;
    }

    /// Builder-style method to set background color
    pub fn with_background(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    /// Builder-style method to set ambient light
    pub fn with_ambient_light(mut self, intensity: Color) -> Self {
        self.ambient_light_intensity = intensity;
        self
    }

    /// Builder-style method to add a surface
    pub fn with_surface(mut self, surface: Box<dyn Surface>) -> Self {
        self.surfaces.push(surface);
        self
    }

    /// Builder-style method to add a light
    pub fn with_light(mut self, light: Light) -> Self {
        self.light_sources.push(light);
        self
    }
}

/// Infinite plane defined by a point and normal
pub struct Plane {
    pub point: Vector3,
    pub normal: Vector3,
    pub material: Material,
}

impl Plane {
    /// Create a new plane
    pub fn new(point: Vector3, normal: Vector3, material: Material) -> Self {
        Self {
            point,
            normal: normal.normalized(),
            material,
        }
    }
}

impl Surface for Plane {
    fn normal(&self, _point: Vector3) -> Vector3 {
        self.normal
    }

    fn material(&self) -> Material {
        self.material
    }
}

impl Render for Plane {
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool {
        let normal_dot_direction = self.normal.dot(&ray.direction);

        // Check if ray is parallel to plane
        if normal_dot_direction.abs() < f32::EPSILON {
            return false;
        }

        let t = (self.point - ray.position).dot(&self.normal) / normal_dot_direction;

        if t < search_interval.0 || t >= search_interval.1 {
            return false;
        }

        hit_rec.t = t;
        hit_rec.normal = self.normal;
        hit_rec.material = self.material();
        true
    }
}

// TODO: pub struct Polygon

// TODO: impl Surface for Polygon

// TODO: impl Render for Polygon
