use crate::debug::is_verbose_enabled;
use crate::primitives::*;
use crate::types::image::*;
use crate::types::linalg::*;
use crate::verbose_println;
use std::sync::Arc;

/// Small positive bias to avoid surface self-intersection artifacts
pub const RAY_BIAS_BASE: Scalar = 0.001;
// Search window limits for ray intersections
pub const MIN_SEARCH: Scalar = RAY_BIAS_BASE;
pub const MAX_SEARCH: Scalar = Scalar::INFINITY;

/// Parametric ray
#[derive(Default, Copy, Clone)]
pub struct Ray {
    pub position: Vector3,
    pub direction: Vector3,
}

/// Represents a camera in 3D space
#[derive(Default, Copy, Clone)]
pub struct Camera {
    pub position: Vector3,
    /// Orthonormal right-handed camera space basis (u, v, w) where -w is the view direction
    pub basis: Basis3,
    /// Pixel resolution as (horizontal, vertical)
    /// Note: Both dimensions must be even
    pub resolution: (usize, usize),
    /// Euclidean distance from camera origin to image plane
    pub focal_length: Scalar,
}

impl Camera {
    pub fn new(
        position: Vector3,
        target: Vector3,
        up: Vector3,
        fov: Scalar,
        resolution: (usize, usize),
    ) -> Self {
        assert!(
            resolution.0 % 2 == 0 && resolution.1 % 2 == 0,
            "Resolution must be even"
        );

        let w = (position - target).normalized(); // Camera looks down -w
        let u = up.cross(&w).normalized(); // Right vector
        let v = w.cross(&u); // Up vector (already normalized)

        let image_plane_width = 1.0;
        let focal_length = (image_plane_width / 2.0) / (fov.to_radians() / 2.0).tan();
        verbose_println!(
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

    /// Takes pixel coordinates and returns coordinates in (u, v) basis of camera space
    /// Note: The (0, 0) pixel is in the top-left corner
    /// Note: Fractional pixel coordinates are supported for subpixel sampling
    pub fn pixel_to_camera_space(&self, pixel_idx: (Scalar, Scalar)) -> Vector2 {
        let x_size = self.resolution.0 as Scalar;
        let y_size = self.resolution.1 as Scalar;
        let aspect_ratio = x_size / y_size;
        let (i, j) = pixel_idx;

        // Image plane spans [-0.5, 0.5] in both dimensions (scaled by aspect ratio)
        let l = -0.5 * aspect_ratio;
        let r = 0.5 * aspect_ratio;
        let t = 0.5;
        let b = -0.5;

        let u = l + (r - l) * (i + 0.5) / x_size;
        let v = t + (b - t) * (j + 0.5) / y_size;
        Vector2(u, v)
    }

    /// Returns view ray for given pixel coordinates
    /// Note: Fractional pixel coordinates are supported for subpixel sampling
    pub fn pixel_to_ray(&self, pixel_idx: (Scalar, Scalar)) -> Ray {
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

/// Represents properties of a surface for rendering
#[derive(Default, Copy, Clone)]
pub struct Material {
    /// Shading shininess factor; non-linear:
    /// 10 = matte, 100 = mildly shiny, 1000 = really glossy, 10000 = mirror
    pub shininess: Scalar,
    /// Main color of the material when it is lit by direct light
    pub diffuse_color: Color,
    /// Color of the shiny highlights on the material when it reflects direct light
    pub specular_color: Color,
    /// Color of the material when it is lit by ambient (indirect) light
    pub ambient_color: Color,
    /// Represents the strength/tint of reflections off the material; Color::BLACK means no reflections
    pub mirror_color: Color,
}

impl Material {
    pub fn new(
        shininess: Scalar,
        diffuse_color: Color,
        specular_color: Color,
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

    /// Create a simple matte material with color
    /// Note: Matte materials have no specular highlights and low shininess
    pub fn matte(color: Color) -> Self {
        Self {
            shininess: 10.0,
            diffuse_color: color,
            specular_color: Color::BLACK,
            ambient_color: color,
            mirror_color: Color::BLACK,
        }
    }

    /// Create a shiny material with color
    pub fn shiny(color: Color) -> Self {
        Self {
            shininess: 100.0,
            diffuse_color: color,
            specular_color: Color::WHITE * 0.3,
            ambient_color: color,
            mirror_color: color * 0.1,
        }
    }

    /// Create a mirror material with scalar reflectance [0.0, 1.0]
    /// Note: Mirror materials have very high shininess and no diffuse/ambient color
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
#[derive(Default)]
pub struct HitRecord {
    /// The surface hit
    // pub surface: &dyn Surface,
    /// Ray t value of intersection
    pub t: Scalar,
    /// Surface normal at intersection point
    pub normal: Vector3,
    /// Material of the surface intersected
    pub material: Material,
}

/// Trait for defining entity rendering behavior
pub trait Render {
    /// Returns the t value of intersection or None if no intersection
    /// Accepts ray, search interval as [t_near, t_far), and a hit record
    fn hit(&self, ray: Ray, search_interval: (Scalar, Scalar), hit_rec: &mut HitRecord) -> bool;
}

/// Calculates adaptive bias to prevent ray self-intersection
fn calculate_ray_bias(intersection_distance: Scalar) -> Scalar {
    RAY_BIAS_BASE * (1.0 + intersection_distance * 0.01)
}

/// Gets the color seen by a ray in a scene
fn raycolor(
    scene: &Scene,
    ray: Ray,
    search_interval: (Scalar, Scalar),
    depth: usize,
    max_depth: usize,
) -> Color {
    verbose_println!("Getting ray color at depth {depth}...");
    if depth > max_depth {
        verbose_println!("Max recursion depth reached, returning background color");
        return scene.background_color;
    }

    let mut hit_rec = HitRecord::default();
    let mut shadow_rec = HitRecord::default();
    if scene.hit(ray, search_interval, &mut hit_rec) {
        let p_intersect = ray.position + (ray.direction * hit_rec.t);
        verbose_println!(
            "Ray hit a surface at {p_intersect:?} with n = {normal:?}! Shading...",
            normal = hit_rec.normal
        );
        let material = hit_rec.material;

        // Add base color due to ambient lighting
        let mut color = material.ambient_color * scene.ambient_light_intensity;
        verbose_println!("Color initial (just ambient lighting) = {color:?}");
        for (light_idx, light) in scene.light_sources.iter().enumerate() {
            verbose_println!(
                "Adding light {light_idx} at position {light_pos:?}...",
                light_pos = light.position
            );
            let light_direction = light.position - p_intersect;
            let light_distance = light_direction.len();
            let light_direction_normalized = light_direction.normalized();

            verbose_println!("Light direction = {light_direction:?}");
            verbose_println!("Normal direction = {normal:?}", normal = hit_rec.normal);

            // Early exit if light is behind surface
            let n_dot_l = hit_rec.normal.dot(&light_direction_normalized);
            if n_dot_l <= 0.0 {
                verbose_println!("Light is behind surface, skipping...");
                continue;
            }

            let shadow_ray = Ray {
                position: p_intersect + hit_rec.normal * calculate_ray_bias(hit_rec.t),
                direction: light_direction_normalized,
            };

            // Check if point is shadowed from this light
            let shadow_interval = (RAY_BIAS_BASE, light_distance - RAY_BIAS_BASE);
            if !scene.hit(shadow_ray, shadow_interval, &mut shadow_rec) {
                verbose_println!("Point is NOT in shadow relative to this light, shading...");

                // Lambertian (diffuse) shading
                color += material.diffuse_color * light.intensity * n_dot_l;
                verbose_println!("Color after Lambertian shading = {color:?}");

                // Blinn-Phong (specular) shading
                let view_direction = ray.direction.normalized() * -1.0;
                let half_vec = (light_direction_normalized + view_direction).normalized();
                let n_dot_h = hit_rec.normal.dot(&half_vec).max(0.0);
                if n_dot_h > 0.0 {
                    color += material.specular_color
                        * light.intensity
                        * n_dot_h.powf(material.shininess);
                }
                verbose_println!("Color after Blinn-Phong shading = {color:?}");
            } else {
                verbose_println!("Point IS in shadow relative to this light, not shading");
            }
        }

        // Get mirror reflection contribution
        if material.mirror_color != Color::BLACK {
            let reflect_ray = Ray {
                position: p_intersect + hit_rec.normal * calculate_ray_bias(hit_rec.t),
                direction: ray.direction
                    - hit_rec.normal * hit_rec.normal.dot(&ray.direction) * 2.0,
            };
            verbose_println!("Hit a mirror, reflecting...");
            color += material.mirror_color
                * raycolor(scene, reflect_ray, search_interval, depth + 1, max_depth);
        }
        verbose_println!("Color after mirror shading = {color:?}");
        color
    } else {
        verbose_println!("Hit nothing, setting to background color...");
        scene.background_color
    }
}

fn shade_pixel(
    x: usize,
    y: usize,
    camera: &Camera,
    scene: &Scene,
    max_depth: usize,
    samples_per_pixel: usize,
    rng: &mut impl rand::Rng,
) -> Pixel {
    verbose_println!("Shading pixel ({x},{y})");
    let mut color = Color::BLACK;
    if samples_per_pixel == 1 {
        // Deterministic single sample per pixel
        let ray = camera.pixel_to_ray((x as Scalar, y as Scalar));
        color = raycolor(scene, ray, (MIN_SEARCH, MAX_SEARCH), 0, max_depth);
    } else {
        // Sample randomly within the pixel for anti-aliasing
        for _ in 0..samples_per_pixel {
            let dx: Scalar = rng.gen::<Scalar>() - 0.5;
            let dy: Scalar = rng.gen::<Scalar>() - 0.5;
            let ray = camera.pixel_to_ray((x as Scalar + dx, y as Scalar + dy));
            color += raycolor(scene, ray, (MIN_SEARCH, MAX_SEARCH), 0, max_depth)
                * (1.0 / samples_per_pixel as Scalar);
        }
    }
    if is_verbose_enabled() {
        let (r8, g8, b8) = color.to_rgb8();
        let (sr8, sg8, sb8) = color.to_srgb8();
        println!(
            "Final pixel ({x},{y}) color: {color:?}, rgb8=({r8},{g8},{b8}), srgb8=({sr8},{sg8},{sb8})",
        );
    }
    Pixel(color)
}

/// Configuration for rendering operations
#[derive(Clone)]
pub struct RenderConfig {
    pub camera: Camera,
    pub scene: Scene,
    pub max_depth: usize,
    pub num_threads: usize,
    pub samples_per_pixel: usize,
    pub cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    pub progress_callback: Option<Arc<dyn Fn(f32) + Send + Sync>>,
}

impl RenderConfig {
    /// Creates a new render configuration
    pub fn new(
        camera: Camera,
        scene: Scene,
        max_depth: usize,
        num_threads: usize,
        samples_per_pixel: usize,
    ) -> Self {
        Self {
            camera,
            scene,
            max_depth,
            num_threads,
            samples_per_pixel,
            cancel_flag: None,
            progress_callback: None,
        }
    }

    /// Creates a basic render configuration from camera and scene
    pub fn basic(camera: Camera, scene: Scene) -> Self {
        Self::new(camera, scene, 3, 8, 4)
    }

    /// Sets the maximum recursion depth for reflections
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the number of threads to use for rendering
    pub fn with_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = num_threads;
        self
    }

    /// Sets the number of samples per pixel for anti-aliasing
    pub fn with_samples_per_pixel(mut self, samples_per_pixel: usize) -> Self {
        self.samples_per_pixel = samples_per_pixel;
        self
    }

    /// Adds cancellation support to the render configuration
    pub fn with_cancellation(mut self, cancel_flag: Arc<std::sync::atomic::AtomicBool>) -> Self {
        self.cancel_flag = Some(cancel_flag);
        self
    }

    /// Adds progress callback to the render configuration
    pub fn with_progress_callback(
        mut self,
        callback: impl Fn(f32) + Send + Sync + 'static,
    ) -> Self {
        self.progress_callback = Some(Arc::new(callback));
        self
    }

    /// Renders the image using this configuration
    pub fn render(self) -> Image {
        render(self)
    }
}

/// Internal configuration for rendering rows
struct RowRenderConfig {
    camera: Camera,
    scene: Scene,
    max_depth: usize,
    width: usize,
    samples_per_pixel: usize,
    cancel_flag: Option<Arc<std::sync::atomic::AtomicBool>>,
    progress_callback: Option<Arc<dyn Fn(usize) + Send + Sync>>,
}

/// Renders a range of rows for the image, returning a vector of (row index, pixel row)
fn render_rows(
    config: RowRenderConfig,
    row_range: std::ops::Range<usize>,
) -> Vec<(usize, Vec<Pixel>)> {
    let mut rows = Vec::new();
    let mut rng = rand::thread_rng();
    for y in row_range {
        if let Some(ref flag) = config.cancel_flag {
            if flag.load(std::sync::atomic::Ordering::Relaxed) {
                verbose_println!("Render cancelled at row {y}");
                break;
            }
        }

        let mut row = Vec::with_capacity(config.width);
        for x in 0..config.width {
            row.push(shade_pixel(
                x,
                y,
                &config.camera,
                &config.scene,
                config.max_depth,
                config.samples_per_pixel,
                &mut rng,
            ));
        }
        rows.push((y, row));

        if let Some(ref callback) = config.progress_callback {
            callback(1);
        }
    }
    rows
}

/// Renders an image using the provided configuration
pub fn render(config: RenderConfig) -> Image {
    let (width, height) = config.camera.resolution;
    let mut frame = Image::new(width, height);

    let total_rows = height;
    let completed_rows = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let row_progress_callback = if let Some(callback) = &config.progress_callback {
        let completed_rows = completed_rows.clone();
        let callback = callback.clone();
        Some(Arc::new(move |rows_completed: usize| {
            let total_completed = completed_rows
                .fetch_add(rows_completed, std::sync::atomic::Ordering::Relaxed)
                + rows_completed;
            let progress = total_completed as f32 / total_rows as f32;
            callback(progress.min(1.0));
        }) as Arc<dyn Fn(usize) + Send + Sync>)
    } else {
        None
    };

    if config.num_threads <= 1 {
        let row_config = RowRenderConfig {
            camera: config.camera,
            scene: config.scene.clone(),
            max_depth: config.max_depth,
            width,
            samples_per_pixel: config.samples_per_pixel,
            cancel_flag: config.cancel_flag.clone(),
            progress_callback: row_progress_callback,
        };
        for (y, row) in render_rows(row_config, 0..height) {
            for (x, pixel) in row.into_iter().enumerate() {
                frame[[x, y]] = pixel;
            }
        }
    } else {
        let rows_per_thread = height.div_ceil(config.num_threads);
        let mut handles = Vec::new();
        for thread_id in 0..config.num_threads {
            let start_row = thread_id * rows_per_thread;
            let end_row = ((thread_id + 1) * rows_per_thread).min(height);
            let camera = config.camera;
            let scene = config.scene.clone();
            let cancel_flag = config.cancel_flag.clone();
            let row_progress_callback = row_progress_callback.clone();
            let max_depth = config.max_depth;
            let samples_per_pixel = config.samples_per_pixel;
            let handle = std::thread::spawn(move || {
                let row_config = RowRenderConfig {
                    camera,
                    scene,
                    max_depth,
                    width,
                    samples_per_pixel,
                    cancel_flag,
                    progress_callback: row_progress_callback,
                };
                render_rows(row_config, start_row..end_row)
            });
            handles.push(handle);
        }
        for handle in handles {
            for (y, row) in handle.join().unwrap() {
                for (x, pixel) in row.into_iter().enumerate() {
                    frame[[x, y]] = pixel;
                }
            }
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
        let Vector2(u, v) = camera.pixel_to_camera_space((0.0, 0.0));
        assert!((u - (l + (r - l) * 0.5 / x_size as Scalar)).abs() < 1e-10);
        assert!((v - (t + (b - t) * 0.5 / y_size as Scalar)).abs() < 1e-10);
        // Top-right
        let Vector2(u, v) = camera.pixel_to_camera_space(((x_size as Scalar - 1.0), 0.0));
        assert!((u - (l + (r - l) * (x_size as Scalar - 0.5) / x_size as Scalar)).abs() < 1e-10);
        assert!((v - (t + (b - t) * 0.5 / y_size as Scalar)).abs() < 1e-10);
        // Bottom-left
        let Vector2(u, v) = camera.pixel_to_camera_space((0.0, y_size as Scalar - 1.0));
        assert!((u - (l + (r - l) * 0.5 / x_size as Scalar)).abs() < 1e-10);
        assert!((v - (t + (b - t) * (y_size as Scalar - 0.5) / y_size as Scalar)).abs() < 1e-10);
        // Bottom-right
        let Vector2(u, v) =
            camera.pixel_to_camera_space(((x_size as Scalar - 1.0), y_size as Scalar - 1.0));
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
            id: 0,
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
