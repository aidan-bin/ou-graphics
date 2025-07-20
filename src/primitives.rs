use crate::render::*;
use crate::types::image::*;
use crate::types::linalg::*;

/// Unique identifier for objects in the scene
pub type ObjectId = usize;
/// Unique identifier for lights in the scene
pub type LightId = usize;

/// 3D scene of renderable objects
pub struct Scene {
    pub background_color: Color,
    pub surfaces: Vec<Box<dyn Surface>>,
    /// Note: Light sources themselves are not visible
    pub light_sources: Vec<Light>,
    /// RGB intensity of the ambient light in the scene (a base level of illumination)
    /// Note: This requires the material to have an ambient_color to have an effect
    pub ambient_light_intensity: Color,
    /// Unique identifier for the next surface to be added
    next_object_id: ObjectId,
    /// Unique identifier for the next light to be added
    next_light_id: LightId,
}

impl Render for Scene {
    /// Updates hit record with closest intersection; returns true if a surface was hit
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool {
        let mut closest_hit = false;
        let mut closest_t = search_interval.1; // Start with far limit

        for surface in &self.surfaces {
            let mut temp_hit_rec = HitRecord::default();
            if surface.hit(ray, (search_interval.0, closest_t), &mut temp_hit_rec)
                && temp_hit_rec.t < closest_t
            {
                closest_t = temp_hit_rec.t;
                *hit_rec = temp_hit_rec;
                closest_hit = true;
            }
        }
        closest_hit
    }
}

impl Clone for Scene {
    fn clone(&self) -> Self {
        Scene {
            background_color: self.background_color,
            surfaces: self.surfaces.iter().map(|s| s.box_clone()).collect(),
            light_sources: self.light_sources.clone(),
            ambient_light_intensity: self.ambient_light_intensity,
            next_object_id: self.next_object_id,
            next_light_id: self.next_light_id,
        }
    }
}

#[derive(Clone)]
pub struct Light {
    pub id: LightId,
    pub position: Vector3,
    /// RGB intensity
    pub intensity: Color,
}

impl Light {
    /// Create a new point light with RGB intensity
    pub fn new(position: Vector3, intensity: Color) -> Self {
        Self {
            id: 0, // Will be set by Scene when added
            position,
            intensity,
        }
    }

    /// Create a white light with scalar intensity [0.0, 1.0]
    pub fn white(position: Vector3, intensity: Scalar) -> Self {
        Self {
            id: 0, // Will be set by Scene when added
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

    /// Get the unique identifier for this light
    pub fn id(&self) -> LightId {
        self.id
    }

    /// Set the unique identifier for this light
    pub fn set_id(&mut self, id: LightId) {
        self.id = id;
    }
}

/// Trait for defining entity structure
pub trait Surface: Render + Send + Sync + 'static {
    /// Returns unit normal vector of surface at point
    /// Note: Assumes point is on surface
    fn normal(&self, point: Vector3) -> Vector3;
    /// Returns the material properties of the surface (used for shading)
    fn material(&self) -> Material;
    /// Returns a mutable reference to the material properties of the surface
    fn material_mut(&mut self) -> &mut Material;
    /// Returns the unique identifier for this surface
    fn id(&self) -> ObjectId;
    /// Set the unique identifier for this surface
    fn set_id(&mut self, id: ObjectId);
    /// Returns the position/center of the object
    fn position(&self) -> Vector3;
    /// Set the position/center of the object
    fn set_position(&mut self, position: Vector3);
    /// Returns string description of the type of this surface
    fn surface_type(&self) -> &'static str;
    fn box_clone(&self) -> Box<dyn Surface>;
}

#[derive(Clone)]
pub struct Sphere {
    pub id: ObjectId,
    pub center: Vector3,
    pub radius: Scalar,
    pub material: Material,
}

impl Surface for Sphere {
    fn normal(&self, point: Vector3) -> Vector3 {
        (point - self.center) * (1.0 / self.radius)
    }

    fn material(&self) -> Material {
        self.material
    }

    fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    fn id(&self) -> ObjectId {
        self.id
    }

    fn set_id(&mut self, id: ObjectId) {
        self.id = id;
    }

    fn position(&self) -> Vector3 {
        self.center
    }

    fn set_position(&mut self, position: Vector3) {
        self.center = position;
    }

    fn surface_type(&self) -> &'static str {
        "Sphere"
    }

    fn box_clone(&self) -> Box<dyn Surface> {
        Box::new(self.clone())
    }
}

impl Render for Sphere {
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool {
        let eye_to_center = ray.position - self.center;

        // Quadratic equation coefficients: at^2 + bt + c = 0
        let a = ray.direction.dot(&ray.direction); // Note: Should be 1.0 for normalized rays
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
    pub fn new(center: Vector3, radius: Scalar, material: Material) -> Self {
        Self {
            id: 0, // Will be set by Scene when added
            center,
            radius,
            material,
        }
    }
}

/// Infinite plane defined by a point and normal
#[derive(Clone)]
pub struct Plane {
    pub id: ObjectId,
    pub point: Vector3,
    pub normal: Vector3,
    pub material: Material,
}

impl Plane {
    pub fn new(point: Vector3, normal: Vector3, material: Material) -> Self {
        Self {
            id: 0, // Will be set by Scene when added
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

    fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    fn id(&self) -> ObjectId {
        self.id
    }

    fn set_id(&mut self, id: ObjectId) {
        self.id = id;
    }

    fn position(&self) -> Vector3 {
        self.point
    }

    fn set_position(&mut self, position: Vector3) {
        self.point = position;
    }

    fn surface_type(&self) -> &'static str {
        "Plane"
    }

    fn box_clone(&self) -> Box<dyn Surface> {
        Box::new(self.clone())
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

/// Convex polygon defined by a list of vertices (in order)
#[derive(Clone)]
pub struct Polygon {
    pub id: ObjectId,
    pub vertices: Vec<Vector3>,
    pub normal: Vector3,
    pub material: Material,
}

impl Polygon {
    /// Create a new convex polygon from vertices (must be coplanar and ordered)
    pub fn new(vertices: Vec<Vector3>, material: Material) -> Self {
        assert!(vertices.len() >= 3, "Polygon must have at least 3 vertices");
        // Compute normal from first three vertices
        let normal = (vertices[1] - vertices[0])
            .cross(&(vertices[2] - vertices[0]))
            .normalized();
        Self {
            id: 0, // Will be set by Scene when added
            vertices,
            normal,
            material,
        }
    }
}

impl Surface for Polygon {
    fn normal(&self, _point: Vector3) -> Vector3 {
        self.normal
    }

    fn material(&self) -> Material {
        self.material
    }

    fn material_mut(&mut self) -> &mut Material {
        &mut self.material
    }

    fn id(&self) -> ObjectId {
        self.id
    }

    fn set_id(&mut self, id: ObjectId) {
        self.id = id;
    }

    fn position(&self) -> Vector3 {
        // Return centroid of vertices
        let mut center = Vector3::ORIGIN;
        for vertex in &self.vertices {
            center = center + *vertex;
        }
        center * (1.0 / self.vertices.len() as Scalar)
    }

    fn set_position(&mut self, position: Vector3) {
        let current_center = self.position();
        let offset = position - current_center;
        for vertex in &mut self.vertices {
            *vertex = *vertex + offset;
        }
    }

    fn surface_type(&self) -> &'static str {
        "Polygon"
    }

    fn box_clone(&self) -> Box<dyn Surface> {
        Box::new(self.clone())
    }
}

impl Render for Polygon {
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool {
        // Ray-plane intersection
        let normal_dot_dir = self.normal.dot(&ray.direction);
        if normal_dot_dir.abs() < f32::EPSILON {
            return false;
        }
        let t = (self.vertices[0] - ray.position).dot(&self.normal) / normal_dot_dir;
        if t < search_interval.0 || t >= search_interval.1 {
            return false;
        }
        let p = ray.position + ray.direction * t;
        // Inside-out test for convex polygon
        let n = self.normal;
        let mut inside = true;
        for i in 0..self.vertices.len() {
            let v0 = self.vertices[i];
            let v1 = self.vertices[(i + 1) % self.vertices.len()];
            let edge = v1 - v0;
            let vp = p - v0;
            if edge.cross(&vp).dot(&n) < 0.0 {
                inside = false;
                break;
            }
        }
        if !inside {
            return false;
        }
        hit_rec.t = t;
        hit_rec.normal = self.normal;
        hit_rec.material = self.material();
        true
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            background_color: Color::BLACK,
            surfaces: Vec::new(),
            light_sources: Vec::new(),
            ambient_light_intensity: Color::BLACK,
            next_object_id: 0,
            next_light_id: 0,
        }
    }
}

impl Scene {
    /// Create a new empty scene with default settings
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_surface(&mut self, mut surface: Box<dyn Surface>) -> ObjectId {
        surface.set_id(self.next_object_id);
        let id = self.next_object_id;
        self.next_object_id += 1;
        self.surfaces.push(surface);
        id
    }

    pub fn remove_surface(&mut self, id: ObjectId) -> bool {
        if let Some(pos) = self.surfaces.iter().position(|s| s.id() == id) {
            self.surfaces.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn get_surface_mut(&mut self, id: ObjectId) -> Option<&mut dyn Surface> {
        self.surfaces
            .iter_mut()
            .find(|s| s.id() == id)
            .map(|s| &mut **s)
    }

    pub fn get_surface(&self, id: ObjectId) -> Option<&dyn Surface> {
        self.surfaces.iter().find(|s| s.id() == id).map(|s| &**s)
    }

    pub fn add_light(&mut self, mut light: Light) -> LightId {
        light.set_id(self.next_light_id);
        let id = self.next_light_id;
        self.next_light_id += 1;
        self.light_sources.push(light);
        id
    }

    pub fn remove_light(&mut self, id: LightId) -> bool {
        if let Some(pos) = self.light_sources.iter().position(|l| l.id() == id) {
            self.light_sources.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn get_light_mut(&mut self, id: LightId) -> Option<&mut Light> {
        self.light_sources.iter_mut().find(|l| l.id() == id)
    }

    pub fn get_light(&self, id: LightId) -> Option<&Light> {
        self.light_sources.iter().find(|l| l.id() == id)
    }

    pub fn set_background(&mut self, color: Color) {
        self.background_color = color;
    }

    pub fn set_ambient_light(&mut self, intensity: Color) {
        self.ambient_light_intensity = intensity;
    }

    pub fn with_background(mut self, color: Color) -> Self {
        self.background_color = color;
        self
    }

    pub fn with_ambient_light(mut self, intensity: Color) -> Self {
        self.ambient_light_intensity = intensity;
        self
    }

    pub fn with_surface(mut self, mut surface: Box<dyn Surface>) -> Self {
        surface.set_id(self.next_object_id);
        self.next_object_id += 1;
        self.surfaces.push(surface);
        self
    }

    pub fn with_light(mut self, mut light: Light) -> Self {
        light.set_id(self.next_light_id);
        self.next_light_id += 1;
        self.light_sources.push(light);
        self
    }
}
