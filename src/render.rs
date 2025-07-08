use crate::debug_println;
use crate::primitives::*;
use crate::types::image::*;
use crate::types::linalg::*;

/// Small positive bias to avoid surface self-intersection artifacts
pub const RAY_BIAS: Scalar = 0.000001;
/// Search window limits for ray intersections
pub const MIN_SEARCH: Scalar = RAY_BIAS;
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
    /// Create a new camera looking at a target point
    pub fn new(
        position: Vector3,
        target: Vector3,
        up: Vector3,
        fov: Scalar,
        resolution: (usize, usize),
    ) -> Self {
        let w = (position - target).normalized(); // Camera looks down -w
        let u = up.cross(&w).normalized(); // Right vector
        let v = w.cross(&u); // Up vector (already normalized)

        let image_plane_width = 1.0;
        let focal_length = (image_plane_width / 2.0) / (fov.to_radians() / 2.0).tan();
        debug_println!(
            "Creating camera at {position:?} looking at {target:?} with up {up:?}, fov {fov}, resolution {resolution:?}, focal length {focal_length}",
        );

        Self {
            position,
            basis: Basis3(u, v, w),
            resolution,
            focal_length,
        }
    }

    pub fn view_direction(&self) -> Vector3 {
        self.basis.2 * -1.0
    }

    /// Takes pixel coordinates and returns coordinates in {u, v} basis of camera space
    /// The (0, 0) pixel is in the top-left corner.
    pub fn pixel_to_camera_space(&self, pixel_idx: (usize, usize)) -> Vector2 {
        let x_size = self.resolution.0 as Scalar;
        let y_size = self.resolution.1 as Scalar;
        let aspect_ratio = x_size / y_size;
        let (i, j) = pixel_idx;

        // Image plane spans [-0.5, 0.5] in both dimensions (scaled by aspect ratio)
        let l = -0.5 * aspect_ratio;
        let r = 0.5 * aspect_ratio;
        let t = 0.5;
        let b = -0.5;

        let u = l + (r - l) * (i as Scalar + 0.5) / x_size;
        let v = t + (b - t) * (j as Scalar + 0.5) / y_size;
        Vector2(u, v)
    }

    /// Returns view ray for given pixel coordinates.
    pub fn pixel_to_ray(&self, pixel_idx: (usize, usize)) -> Ray {
        let Vector2(u, v) = self.pixel_to_camera_space((pixel_idx.0, pixel_idx.1));
        let view_vec = self.view_direction();
        let u_vec = self.basis.0;
        let v_vec = self.basis.1;
        let e = self.position;
        let d = ((view_vec * self.focal_length) + (u_vec * u) + (v_vec * v)).normalized();
        Ray {
            position: e,
            direction: d,
        }
    }
}

/// Struct modeling apparent properties of a surface
#[derive(Default, Copy, Clone)]
pub struct Material {
    pub shininess: Scalar, // 10 = matte, 100 = mildly shiny, 1000 = really glossy, 10000 = mirror
    /// Surface color
    pub diffuse_color: Color, // Main color of the material when it is lit by direct light
    pub specular_color: Color, // Color of the shiny highlights that appear on the material when it reflects light sources
    pub ambient_color: Color, // Color of the material under ambient (indirect) light, representing how it looks in shadow or low light
    pub mirror_color: Color, // Color of the light reflected by the material as if it were a mirror; used to simulate mirror-like reflections
}

impl Material {
    /// Create a new material with all properties
    pub fn new(
        diffuse_color: Color,
        specular_color: Color,
        shininess: Scalar,
        ambient_color: Color,
        mirror_color: Color,
    ) -> Self {
        Self {
            shininess,
            diffuse_color,
            specular_color,
            ambient_color,
            mirror_color,
        }
    }

    /// Create a simple matte material
    pub fn matte(color: Color) -> Self {
        Self {
            shininess: 10.0,
            diffuse_color: color,
            specular_color: Color::BLACK,
            ambient_color: color * 0.2,
            mirror_color: Color::BLACK,
        }
    }

    /// Create a shiny material
    pub fn shiny(color: Color, shininess: Scalar) -> Self {
        Self {
            shininess,
            diffuse_color: color,
            specular_color: Color::WHITE * 0.3,
            ambient_color: color * 0.1,
            mirror_color: Color::BLACK,
        }
    }

    /// Create a mirror material
    pub fn mirror(reflectance: Scalar) -> Self {
        let mirror_color = Color::WHITE * reflectance;
        Self {
            shininess: 1000.0,
            diffuse_color: Color::BLACK,
            specular_color: Color::WHITE,
            ambient_color: Color::BLACK,
            mirror_color,
        }
    }
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
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool;
}

/// Gets the color seen by a ray in a scene.
fn raycolor(
    scene: &Scene,
    ray: Ray,
    search_interval: (Scalar, Scalar),
    depth: usize,
    max_depth: usize,
) -> Color {
    debug_println!("Getting ray color at depth {depth}...");
    if depth > max_depth {
        debug_println!("Max recursion depth reached, returning background color");
        return scene.background_color;
    }

    let mut hit_rec = HitRecord::default();
    let mut shadow_rec = HitRecord::default();
    if scene.hit(ray, search_interval, &mut hit_rec) {
        let p_intersect = ray.position + (ray.direction * hit_rec.t);
        debug_println!(
            "Ray hit a surface at {p_intersect:?} with n = {normal:?}! Shading...",
            normal = hit_rec.normal
        );
        let material = hit_rec.material;

        // Add base color due to ambient lighting
        let mut color = material.ambient_color * scene.ambient_light_intensity;
        debug_println!("Color initial (just ambient lighting) = {color:?}");
        for (light_idx, light) in scene.light_sources.iter().enumerate() {
            debug_println!("Adding light {light_idx}...");
            let light_direction = light.position - p_intersect;
            let light_distance = light_direction.len();
            let light_direction_normalized = light_direction.normalized();

            debug_println!("Light direction = {light_direction:?}");
            debug_println!("Normal direction = {normal:?}", normal = hit_rec.normal);

            // Early exit if light is behind surface
            let n_dot_l = hit_rec.normal.dot(&light_direction_normalized);
            if n_dot_l <= 0.0 {
                debug_println!("Light is behind surface, skipping...");
                continue;
            }

            let shadow_ray = Ray {
                position: p_intersect + hit_rec.normal * RAY_BIAS,
                direction: light_direction_normalized,
            };

            let shadow_interval = (RAY_BIAS, light_distance - RAY_BIAS);
            if !scene.hit(shadow_ray, shadow_interval, &mut shadow_rec) {
                debug_println!("Point is NOT in shadow relative to this light, shading...");

                // Lambertian (diffuse) shading
                color += material.diffuse_color * light.intensity * n_dot_l;
                debug_println!("Color after Lambertian shading = {color:?}");

                // Blinn-Phong (specular) shading
                let view_direction = ray.direction.normalized() * -1.0;
                let half_vec = (light_direction_normalized + view_direction).normalized();
                let n_dot_h = hit_rec.normal.dot(&half_vec).max(0.0);
                if n_dot_h > 0.0 {
                    color += material.specular_color
                        * light.intensity
                        * n_dot_h.powf(material.shininess);
                }
                debug_println!("Color after Blinn-Phong shading = {color:?}");
            } else {
                debug_println!("Point IS in shadow relative to this light, not shading");
            }
        }

        // Get mirror reflection contribution
        if material.mirror_color != Color::BLACK {
            let reflect_ray = Ray {
                position: p_intersect + hit_rec.normal * RAY_BIAS,
                direction: ray.direction
                    - hit_rec.normal * hit_rec.normal.dot(&ray.direction) * 2.0,
            };
            debug_println!("Hit a mirror, reflecting...");
            color += material.mirror_color
                * raycolor(scene, reflect_ray, search_interval, depth + 1, max_depth);
        }
        debug_println!("Color after mirror shading = {color:?}");
        color
    } else {
        debug_println!("Hit nothing, setting to background color...");
        scene.background_color
    }
}

pub fn render(camera: &Camera, scene: &Scene, max_depth: usize) -> Image {
    let x_size = camera.resolution.0;
    let y_size = camera.resolution.1;
    let mut frame = Image::new(x_size, y_size);
    for i in 0..x_size {
        for j in 0..y_size {
            debug_println!("Shading pixel ({i},{j})");
            let ray = camera.pixel_to_ray((i, j));
            let search_interval = (MIN_SEARCH, MAX_SEARCH);
            frame[[i, j]] = Pixel(raycolor(scene, ray, search_interval, 0, max_depth));
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use crate::render::*;

    #[test]
    fn camera_space_conversion_fits_corners() {
        let x_size = 4;
        let y_size = 4;
        let aspect_ratio = x_size as Scalar / y_size as Scalar;

        // Expected bounds
        let l = -0.5 * aspect_ratio;
        let r = 0.5 * aspect_ratio;
        let t = 0.5;
        let b = -0.5;

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
        let Vector2(u, v) = camera.pixel_to_camera_space((0, 0));
        assert!((u - (l + (r - l) * 0.5 / x_size as Scalar)).abs() < 1e-10);
        assert!((v - (t + (b - t) * 0.5 / y_size as Scalar)).abs() < 1e-10);
        // Top-right
        let Vector2(u, v) = camera.pixel_to_camera_space((x_size - 1, 0));
        assert!((u - (l + (r - l) * (x_size as Scalar - 0.5) / x_size as Scalar)).abs() < 1e-10);
        assert!((v - (t + (b - t) * 0.5 / y_size as Scalar)).abs() < 1e-10);
        // Bottom-left
        let Vector2(u, v) = camera.pixel_to_camera_space((0, y_size - 1));
        assert!((u - (l + (r - l) * 0.5 / x_size as Scalar)).abs() < 1e-10);
        assert!((v - (t + (b - t) * (y_size as Scalar - 0.5) / y_size as Scalar)).abs() < 1e-10);
        // Bottom-right
        let Vector2(u, v) = camera.pixel_to_camera_space((x_size - 1, y_size - 1));
        assert!((u - (l + (r - l) * (x_size as Scalar - 0.5) / x_size as Scalar)).abs() < 1e-10);
        assert!((v - (t + (b - t) * (y_size as Scalar - 0.5) / y_size as Scalar)).abs() < 1e-10);
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
