use eframe::egui;
use ou_graphics::primitives::*;
use ou_graphics::render::*;
use ou_graphics::types::image::*;
use ou_graphics::types::linalg::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Configuration constants for the ray tracer GUI
mod config {
    use std::time::Duration;

    pub const DEFAULT_RENDER_RESOLUTION: u32 = 512;
    pub const MIN_RENDER_RESOLUTION: u32 = 128;
    pub const MAX_RENDER_RESOLUTION: u32 = 2048;
    pub const RESOLUTION_DEBOUNCE_DELAY: Duration = Duration::from_millis(300);
    pub const DISPLAY_IMAGE_SIZE: f32 = 800.0;
    pub const DOWNSAMPLING_THRESHOLD: u32 = 1024;

    pub const DEFAULT_CAMERA_DISTANCE: f32 = 5.0;
    pub const DEFAULT_SAMPLES_PER_PIXEL: usize = 4;
    pub const DEFAULT_MAX_DEPTH: usize = 3;
    pub const DEFAULT_THREAD_COUNT: usize = 8;

    pub const MIN_CAMERA_DISTANCE: f32 = 1.0;
    pub const MAX_CAMERA_DISTANCE: f32 = 20.0;
    pub const MIN_CAMERA_ANGLE_V: f32 = -89.0;
    pub const MAX_CAMERA_ANGLE_V: f32 = 89.0;
    pub const MIN_CAMERA_ANGLE_H: f32 = -180.0;
    pub const MAX_CAMERA_ANGLE_H: f32 = 180.0;

    pub const RGBA_CHANNELS: u32 = 4;
}

/// Manages the camera state and controls
#[derive(Debug, Clone)]
struct CameraController {
    distance: f32,
    angle_horizontal: f32,
    angle_vertical: f32,
}

impl Default for CameraController {
    fn default() -> Self {
        Self {
            distance: config::DEFAULT_CAMERA_DISTANCE,
            angle_horizontal: 0.0,
            angle_vertical: 0.0,
        }
    }
}

impl CameraController {
    /// Creates a camera from the current controller state
    fn create_camera(&self, resolution: (usize, usize)) -> Camera {
        let position = self.calculate_position();
        Camera::new(position, Vector3::ORIGIN, Vector3::Z, 60.0, resolution)
    }

    /// Calculates the camera position based on spherical coordinates
    fn calculate_position(&self) -> Vector3 {
        let h_rad = self
            .angle_horizontal
            .clamp(config::MIN_CAMERA_ANGLE_H, config::MAX_CAMERA_ANGLE_H)
            .to_radians();
        let v_rad = self
            .angle_vertical
            .clamp(config::MIN_CAMERA_ANGLE_V, config::MAX_CAMERA_ANGLE_V)
            .to_radians();

        let x = self.distance * h_rad.cos() * v_rad.cos();
        let y = self.distance * h_rad.sin() * v_rad.cos();
        let z = self.distance * v_rad.sin();

        Vector3(x, y, z)
    }

    /// Resets the camera to default position
    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Manages render settings and state
#[derive(Debug, Clone)]
struct RenderSettings {
    samples_per_pixel: usize,
    max_depth: usize,
    thread_count: usize,
    resolution: u32,
    auto_render: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            samples_per_pixel: config::DEFAULT_SAMPLES_PER_PIXEL,
            max_depth: config::DEFAULT_MAX_DEPTH,
            thread_count: config::DEFAULT_THREAD_COUNT,
            resolution: config::DEFAULT_RENDER_RESOLUTION,
            auto_render: true,
        }
    }
}

/// Manages the render thread and its state
struct RenderManager {
    thread_handle: Option<thread::JoinHandle<()>>,
    result_receiver: Option<mpsc::Receiver<Duration>>,
    preview_pixels: Arc<Mutex<Vec<u8>>>,
    in_progress: bool,
    last_render_time: Option<Duration>,
    pending_resolution: Option<u32>,
    last_resolution_change: Option<Instant>,
    cancel_flag: Arc<AtomicBool>,
    render_progress: Arc<Mutex<f32>>,
}

impl Default for RenderManager {
    fn default() -> Self {
        let initial_size = (config::DEFAULT_RENDER_RESOLUTION
            * config::DEFAULT_RENDER_RESOLUTION
            * config::RGBA_CHANNELS) as usize;
        Self {
            thread_handle: None,
            result_receiver: None,
            preview_pixels: Arc::new(Mutex::new(vec![0; initial_size])),
            in_progress: false,
            last_render_time: None,
            pending_resolution: None,
            last_resolution_change: None,
            cancel_flag: Arc::new(AtomicBool::new(false)),
            render_progress: Arc::new(Mutex::new(0.0)),
        }
    }
}

impl RenderManager {
    /// Starts a new render in the background
    fn start_render(
        &mut self,
        camera: Camera,
        scene: &Scene,
        settings: &RenderSettings,
    ) -> Result<(), String> {
        // Cancel any existing render and clean up
        self.cancel_current_render();

        self.in_progress = true;
        self.cancel_flag.store(false, Ordering::Relaxed);

        if let Ok(mut progress) = self.render_progress.lock() {
            *progress = 0.0;
        }

        // Ensure pixel buffer is the right size
        if let Err(e) = self.resize_pixel_buffer(settings.resolution) {
            return Err(format!("Failed to resize pixel buffer: {e}"));
        }

        // Clone data for background thread
        let scene = scene.clone();
        let max_depth = settings.max_depth;
        let samples = settings.samples_per_pixel;
        let thread_count = settings.thread_count;
        let pixels_arc = self.preview_pixels.clone();
        let render_resolution = settings.resolution;
        let cancel_flag = self.cancel_flag.clone();
        let progress_arc = self.render_progress.clone();

        // Create communication channel
        let (sender, receiver) = mpsc::channel();
        self.result_receiver = Some(receiver);

        // Spawn render thread
        let handle = thread::spawn(move || {
            let start = Instant::now();

            let progress_callback = move |progress: f32| {
                if let Ok(mut p) = progress_arc.lock() {
                    *p = progress;
                }
            };

            let config = RenderConfig::new(camera, scene, max_depth, thread_count, samples)
                .with_cancellation(cancel_flag.clone())
                .with_progress_callback(progress_callback);

            let frame = render(config);

            if cancel_flag.load(Ordering::Relaxed) {
                let _ = sender.send(start.elapsed());
                return;
            }

            // Update pixel buffer
            match pixels_arc.lock() {
                Ok(mut pixels) => {
                    let total_pixels =
                        (render_resolution * render_resolution * config::RGBA_CHANNELS) as usize;
                    pixels.clear();
                    pixels.reserve_exact(total_pixels);

                    for j in 0..render_resolution {
                        for i in 0..render_resolution {
                            let color = frame[[i as usize, j as usize]].0;
                            let (r, g, b) = color.to_srgb8();
                            pixels.extend_from_slice(&[r, g, b, 255]);
                        }
                    }
                }
                Err(_) => {
                    eprintln!("Error: Pixel buffer mutex is poisoned - cannot update display");
                    return;
                }
            }

            let duration = start.elapsed();
            let _ = sender.send(duration);
        });

        self.thread_handle = Some(handle);
        Ok(())
    }

    /// Checks if render is complete and updates state
    /// Returns true if a render just completed
    fn check_render_completion(&mut self) -> bool {
        if let Some(receiver) = &self.result_receiver {
            if let Ok(duration) = receiver.try_recv() {
                self.last_render_time = Some(duration);

                if let Ok(mut progress) = self.render_progress.lock() {
                    *progress = 1.0;
                }

                // Clean up thread handle (should be finished if we got a result)
                if let Some(handle) = self.thread_handle.take() {
                    if let Err(e) = handle.join() {
                        eprintln!("Warning: Render thread panicked during completion: {e:?}");
                    }
                }

                self._reset_render_state();
                return true;
            }
        }
        false
    }

    /// Handles resolution change with debouncing
    fn handle_resolution_change(&mut self, new_resolution: u32, current_settings_resolution: u32) {
        let effective_current = self
            .pending_resolution
            .unwrap_or(current_settings_resolution);
        if new_resolution != effective_current {
            self.pending_resolution = Some(new_resolution);
            self.last_resolution_change = Some(Instant::now());
        }
    }

    /// Applies pending resolution change if debounce period has elapsed
    fn apply_pending_resolution(&mut self, settings: &mut RenderSettings) -> bool {
        if let (Some(pending_res), Some(change_time)) =
            (self.pending_resolution, self.last_resolution_change)
        {
            if change_time.elapsed() >= config::RESOLUTION_DEBOUNCE_DELAY {
                self.cancel_current_render();

                if let Err(e) = self.resize_pixel_buffer(pending_res) {
                    eprintln!("Error: Failed to resize pixel buffer: {e}");
                    return false;
                }

                settings.resolution = pending_res;
                self.pending_resolution = None;
                self.last_resolution_change = None;
                return true;
            }
        }
        false
    }

    /// Cancels the current render if one is in progress
    /// Attempts graceful cancellation with timeout before forcing cleanup
    fn cancel_current_render(&mut self) {
        if !self.in_progress {
            return;
        }

        self.cancel_flag.store(true, Ordering::Relaxed);

        if let Some(handle) = self.thread_handle.take() {
            // Wait a reasonable time for graceful cancellation
            const GRACEFUL_TIMEOUT: Duration = Duration::from_millis(200);
            let start = Instant::now();

            while !handle.is_finished() && start.elapsed() < GRACEFUL_TIMEOUT {
                std::thread::sleep(Duration::from_millis(1));
            }

            if handle.is_finished() {
                if let Err(e) = handle.join() {
                    eprintln!("Warning: Render thread panicked during cancellation: {e:?}");
                }
            } else {
                eprintln!(
                    "Warning: Render thread didn't respond to cancellation in time, detaching..."
                );
                // Thread will clean up on its own when it checks the cancel flag
                std::mem::forget(handle);
            }
        }

        self._reset_render_state();
    }

    /// Resets the render state to initial values
    fn _reset_render_state(&mut self) {
        self.in_progress = false;
        self.result_receiver = None;

        if let Ok(mut progress) = self.render_progress.lock() {
            *progress = 0.0;
        }
    }

    /// Resizes the pixel buffer for the given resolution
    /// Returns an error if the resolution would cause memory issues
    fn resize_pixel_buffer(&mut self, resolution: u32) -> Result<(), String> {
        // Validate resolution bounds first
        if !(config::MIN_RENDER_RESOLUTION..=config::MAX_RENDER_RESOLUTION).contains(&resolution) {
            return Err(format!(
                "Resolution {resolution} is outside valid range [{}, {}]",
                config::MIN_RENDER_RESOLUTION,
                config::MAX_RENDER_RESOLUTION
            ));
        }

        // Check for potential overflow (only an issue for very large resolutions)
        let area = resolution
            .checked_mul(resolution)
            .ok_or_else(|| format!("Resolution {resolution}x{resolution} causes overflow"))?;

        let size_check = area
            .checked_mul(config::RGBA_CHANNELS)
            .ok_or_else(|| format!("Resolution {resolution}x{resolution} causes overflow"))?;

        let expected_size = size_check as usize;
        if expected_size > usize::MAX / 2 {
            return Err(format!("Resolution {resolution}x{resolution} too large"));
        }

        match self.preview_pixels.lock() {
            Ok(mut pixels) => {
                if pixels.len() != expected_size {
                    pixels.clear();
                    pixels.resize(expected_size, 0);
                }
                Ok(())
            }
            Err(e) => Err(format!("Failed to access pixel buffer: {e}")),
        }
    }

    /// Gets the current render progress (0.0 to 1.0)
    fn get_render_progress(&self) -> f32 {
        self.render_progress.lock().map(|p| *p).unwrap_or(0.0)
    }
}

/// Selection state for scene objects and lights
#[derive(Debug, Default)]
struct SelectionState {
    selected_object_id: Option<ObjectId>,
    selected_light_id: Option<LightId>,
}

impl SelectionState {
    fn clear_object_selection(&mut self) {
        self.selected_object_id = None;
    }

    fn clear_light_selection(&mut self) {
        self.selected_light_id = None;
    }

    fn select_object(&mut self, id: ObjectId) {
        self.selected_object_id = Some(id);
    }

    fn select_light(&mut self, id: LightId) {
        self.selected_light_id = Some(id);
    }
}

/// Factory for creating default scenes
struct SceneFactory;

impl SceneFactory {
    /// Creates a default demo scene with various objects
    fn create_default_scene() -> Scene {
        let ground = Plane::new(
            Vector3(0.0, 0.0, -1.0),
            Vector3::Z,
            Material::shiny(Color::WHITE * 0.7),
        );

        let mirror_ball = Sphere::new(
            Vector3(0.0, 0.0, 0.0),
            1.0,
            Material::new(
                1000.0,
                Color::BLACK,
                Color::WHITE,
                Color::WHITE * 0.2, // Slightly opaque
                Color::WHITE,       // Full mirror
            ),
        );

        let small_balls = vec![
            Sphere::new(Vector3(2.0, 1.5, -0.5), 0.5, Material::shiny(Color::RED)),
            Sphere::new(Vector3(-2.0, -2.0, -0.7), 0.3, Material::shiny(Color::BLUE)),
            Sphere::new(Vector3(1.5, -1.5, -0.8), 0.2, Material::shiny(Color::GREEN)),
            Sphere::new(Vector3(0.8, 2.0, -0.7), 0.3, Material::shiny(Color::CYAN)),
            Sphere::new(
                Vector3(-1.2, 1.2, -0.6),
                0.4,
                Material::shiny(Color::YELLOW),
            ),
        ];

        let sky_color = Color(0.4, 0.7, 1.0) * 0.8;
        let ambient_light = Color::WHITE * 0.3;
        let sun_light = Light::white(Vector3(10.0, 10.0, 20.0), 1.0);

        let mut scene = Scene::default()
            .with_background(sky_color)
            .with_ambient_light(ambient_light)
            .with_surface(Box::new(ground))
            .with_surface(Box::new(mirror_ball))
            .with_light(sun_light);

        for ball in small_balls {
            scene.add_surface(Box::new(ball));
        }

        scene
    }
}

/// Main application state for the ray tracer GUI
pub struct RayTracerApp {
    scene: Scene,
    camera_controller: CameraController,
    render_settings: RenderSettings,
    render_manager: RenderManager,
    selection_state: SelectionState,
    cached_camera: Option<Camera>,
    last_error: Option<String>,
    camera_cache_valid: bool,
}

impl Default for RayTracerApp {
    fn default() -> Self {
        Self::new()
    }
}

impl RayTracerApp {
    /// Creates a new ray tracer application with default scene
    pub fn new() -> Self {
        let scene = SceneFactory::create_default_scene();

        let mut app = Self {
            scene,
            camera_controller: CameraController::default(),
            render_settings: RenderSettings::default(),
            render_manager: RenderManager::default(),
            selection_state: SelectionState::default(),
            cached_camera: None,
            last_error: None,
            camera_cache_valid: false,
        };

        // Initial render
        app.trigger_render();
        app
    }

    /// Updates the cached camera when settings change
    fn update_camera(&mut self) {
        if !self.camera_cache_valid {
            let resolution = (
                self.render_settings.resolution as usize,
                self.render_settings.resolution as usize,
            );
            self.cached_camera = Some(self.camera_controller.create_camera(resolution));
            self.camera_cache_valid = true;
        }
    }

    /// Invalidates the camera cache when camera settings change
    fn invalidate_camera_cache(&mut self) {
        self.camera_cache_valid = false;
    }

    /// Triggers a render if auto-render is enabled
    /// Sets last_error if the render fails to start
    fn trigger_render(&mut self) {
        if !self.render_settings.auto_render {
            return;
        }
        self._do_render();
    }

    /// Forces a render regardless of auto-render setting
    /// Sets last_error if the render fails to start
    fn force_render(&mut self) {
        self._do_render();
    }

    /// Internal helper to perform the actual render operation
    fn _do_render(&mut self) {
        self.update_camera();
        if let Some(camera) = &self.cached_camera {
            if let Err(err) =
                self.render_manager
                    .start_render(*camera, &self.scene, &self.render_settings)
            {
                self.last_error = Some(format!("Render failed: {err}"));
            } else {
                self.last_error = None;
            }
        } else {
            self.last_error = Some("Camera configuration invalid".to_string());
        }
    }

    /// Updates application state (should be called each frame)
    fn update_state(&mut self) {
        self.render_manager.check_render_completion();

        // Apply pending resolution changes
        if self
            .render_manager
            .apply_pending_resolution(&mut self.render_settings)
        {
            self.invalidate_camera_cache();
            self.trigger_render();
        }
    }

    /// Adds a new sphere to the scene
    fn add_sphere(&mut self) {
        let sphere = Sphere::new(Vector3(0.0, 0.0, 0.0), 0.5, Material::shiny(Color::GREEN));
        let id = self.scene.add_surface(Box::new(sphere));
        self.selection_state.select_object(id);
        self.trigger_render();
    }

    /// Adds a new plane to the scene
    fn add_plane(&mut self) {
        let plane = Plane::new(
            Vector3(0.0, 0.0, 0.0),
            Vector3::Z,
            Material::shiny(Color::GRAY),
        );
        let id = self.scene.add_surface(Box::new(plane));
        self.selection_state.select_object(id);
        self.trigger_render();
    }

    /// Adds a new light to the scene
    fn add_light(&mut self) {
        let light = Light::white(Vector3(0.0, 0.0, 0.0), 1.0);
        let id = self.scene.add_light(light);
        self.selection_state.select_light(id);
        self.trigger_render();
    }

    /// Deletes the currently selected object
    fn delete_selected_object(&mut self) {
        if let Some(id) = self.selection_state.selected_object_id {
            if self.scene.remove_surface(id) {
                self.selection_state.clear_object_selection();
                self.trigger_render();
            }
        }
    }

    /// Deletes the currently selected light
    fn delete_selected_light(&mut self) {
        if let Some(light_id) = self.selection_state.selected_light_id {
            if self.scene.remove_light(light_id) {
                self.selection_state.clear_light_selection();
                self.trigger_render();
            }
        }
    }

    /// Renders the left control panel
    fn render_control_panel(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Ray Tracer Controls");

                let camera_changed = ui_components::render_camera_controls(ui, self);
                if camera_changed {
                    self.invalidate_camera_cache();
                    if self.render_settings.auto_render {
                        self.trigger_render();
                    }
                }

                let settings_changed = ui_components::render_settings_controls(ui, self);
                if settings_changed && self.render_settings.auto_render {
                    self.trigger_render();
                }

                ui_components::render_resolution_controls(ui, self);
                self.render_lighting_controls(ui);
                let _objects_changed = ui_components::render_object_controls(ui, self);
                self.render_selected_object_properties(ui);
                self.render_light_controls(ui);
                self.render_status_section(ui);
            });
        });
    }

    /// Renders the main preview panel
    fn render_main_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Preview");
            self.render_preview_panel(ui, ctx);
        });
    }

    /// Renders the status section with render button and progress info
    fn render_status_section(&mut self, ui: &mut egui::Ui) {
        ui.separator();

        ui.horizontal(|ui| {
            if ui.button("Render").clicked() {
                self.force_render();
            }

            if self.render_manager.in_progress && ui.button("Cancel").clicked() {
                self.render_manager.cancel_current_render();
            }
        });

        // Display any errors
        if let Some(error) = &self.last_error {
            ui.colored_label(egui::Color32::RED, format!("⚠ {error}"));
        }

        if let Some(render_time) = self.render_manager.last_render_time {
            ui.label(format!("Last render: {render_time:.2?}"));
        }

        if self.render_manager.in_progress {
            ui.label("🔄 Rendering...");

            let progress = self.render_manager.get_render_progress();
            let progress_bar = egui::ProgressBar::new(progress)
                .text(format!("{:.1}%", progress * 100.0))
                .animate(true);
            ui.add(progress_bar);

            if self.render_settings.samples_per_pixel > 8 {
                ui.label("(High sample count - may take several seconds)");
            }
        }
    }
}

/// UI components and layout management
mod ui_components {
    use super::*;

    /// Renders the camera control section
    pub fn render_camera_controls(ui: &mut egui::Ui, app: &mut RayTracerApp) -> bool {
        let mut camera_changed = false;

        ui.separator();
        ui.label("Camera Controls:");

        ui.horizontal(|ui| {
            ui.label("Distance:");
            if ui
                .add(egui::Slider::new(
                    &mut app.camera_controller.distance,
                    config::MIN_CAMERA_DISTANCE..=config::MAX_CAMERA_DISTANCE,
                ))
                .changed()
            {
                camera_changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Horizontal:");
            if ui
                .add(
                    egui::Slider::new(
                        &mut app.camera_controller.angle_horizontal,
                        config::MIN_CAMERA_ANGLE_H..=config::MAX_CAMERA_ANGLE_H,
                    )
                    .suffix("°"),
                )
                .changed()
            {
                camera_changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Vertical:");
            if ui
                .add(
                    egui::Slider::new(
                        &mut app.camera_controller.angle_vertical,
                        config::MIN_CAMERA_ANGLE_V..=config::MAX_CAMERA_ANGLE_V,
                    )
                    .suffix("°"),
                )
                .changed()
            {
                camera_changed = true;
            }
        });

        if ui.button("Reset Camera").clicked() {
            app.camera_controller.reset();
            camera_changed = true;
        }

        camera_changed
    }

    /// Renders the render settings section
    pub fn render_settings_controls(ui: &mut egui::Ui, app: &mut RayTracerApp) -> bool {
        let mut settings_changed = false;

        ui.separator();
        ui.label("Render Settings:");

        ui.checkbox(&mut app.render_settings.auto_render, "Auto Render");

        ui.horizontal(|ui| {
            ui.label("Samples per Pixel:");
            if ui
                .add(egui::Slider::new(
                    &mut app.render_settings.samples_per_pixel,
                    1..=64,
                ))
                .changed()
            {
                settings_changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Thread Count:");
            if ui
                .add(egui::Slider::new(
                    &mut app.render_settings.thread_count,
                    1..=64,
                ))
                .changed()
            {
                settings_changed = true;
            }
        });

        ui.horizontal(|ui| {
            ui.label("Max Depth:");
            if ui
                .add(egui::Slider::new(
                    &mut app.render_settings.max_depth,
                    1..=10,
                ))
                .changed()
            {
                settings_changed = true;
            }
        });

        settings_changed
    }

    /// Renders the resolution control section
    pub fn render_resolution_controls(ui: &mut egui::Ui, app: &mut RayTracerApp) {
        ui.horizontal(|ui| {
            ui.label("Render Resolution:");
            let current_resolution = if let Some(pending) = app.render_manager.pending_resolution {
                pending as f32
            } else {
                app.render_settings.resolution as f32
            };
            let mut temp_resolution = current_resolution;

            if ui
                .add(
                    egui::Slider::new(
                        &mut temp_resolution,
                        config::MIN_RENDER_RESOLUTION as f32..=config::MAX_RENDER_RESOLUTION as f32,
                    )
                    .suffix("x")
                    .logarithmic(true)
                    .step_by(2.0),
                )
                .changed()
            {
                let new_resolution = (temp_resolution as u32 / 2) * 2;
                app.render_manager
                    .handle_resolution_change(new_resolution, app.render_settings.resolution);
            }

            let display_resolution = app
                .render_manager
                .pending_resolution
                .unwrap_or(app.render_settings.resolution);
            ui.label(format!("{display_resolution}x{display_resolution}"));
        });

        ui.horizontal(|ui| {
            ui.label("Presets:");
            for &resolution in &[256, 512, 1024, 2048] {
                if ui.small_button(resolution.to_string()).clicked() {
                    app.render_manager
                        .handle_resolution_change(resolution, app.render_settings.resolution);
                }
            }
        });
    }

    /// Renders the scene object list and controls
    pub fn render_object_controls(ui: &mut egui::Ui, app: &mut RayTracerApp) -> bool {
        let mut scene_changed = false;

        ui.separator();
        ui.label("Scene Objects:");

        if ui.button("Add Sphere").clicked() {
            app.add_sphere();
            scene_changed = true;
        }

        if ui.button("Add Plane").clicked() {
            app.add_plane();
            scene_changed = true;
        }

        if ui.button("Delete Selected").clicked() {
            app.delete_selected_object();
            scene_changed = true;
        }

        ui.separator();
        ui.label("Objects:");

        for surface in app.scene.surfaces.iter() {
            let id = surface.id();
            let is_selected = app.selection_state.selected_object_id == Some(id);
            let label = format!("{} (ID: {})", surface.surface_type(), id);

            if ui.selectable_label(is_selected, label).clicked() {
                app.selection_state.select_object(id);
            }
        }

        scene_changed
    }
}

// Implementation of the main eframe::App trait
impl eframe::App for RayTracerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_state();
        self.render_control_panel(ctx);
        self.render_main_panel(ctx);
        ctx.request_repaint();
    }
}

// Additional UI methods for RayTracerApp
impl RayTracerApp {
    /// Renders the lighting controls section
    fn render_lighting_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Global Lighting:");

        // Ambient light control
        let mut ambient_color = [
            self.scene.ambient_light_intensity.0,
            self.scene.ambient_light_intensity.1,
            self.scene.ambient_light_intensity.2,
        ];

        ui.horizontal(|ui| {
            ui.label("Ambient Light:");
            if ui.color_edit_button_rgb(&mut ambient_color).changed() {
                self.scene.ambient_light_intensity =
                    Color(ambient_color[0], ambient_color[1], ambient_color[2]);
                self.trigger_render();
            }
        });

        // Background color control
        let mut bg_color = [
            self.scene.background_color.0,
            self.scene.background_color.1,
            self.scene.background_color.2,
        ];

        ui.horizontal(|ui| {
            ui.label("Background:");
            if ui.color_edit_button_rgb(&mut bg_color).changed() {
                self.scene.background_color = Color(bg_color[0], bg_color[1], bg_color[2]);
                self.trigger_render();
            }
        });
    }

    /// Renders the selected object properties section
    fn render_selected_object_properties(&mut self, ui: &mut egui::Ui) {
        if let Some(id) = self.selection_state.selected_object_id {
            ui.separator();
            ui.label("Selected Object Properties:");

            let mut needs_render = false;

            if let Some(surface) = self.scene.get_surface_mut(id) {
                let mut pos = surface.position();
                let mut material = surface.material();

                ui.horizontal(|ui| {
                    ui.label("X:");
                    if ui
                        .add(egui::Slider::new(&mut pos.0, -5.0..=5.0).step_by(0.1))
                        .changed()
                    {
                        needs_render = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Y:");
                    if ui
                        .add(egui::Slider::new(&mut pos.1, -5.0..=5.0).step_by(0.1))
                        .changed()
                    {
                        needs_render = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Z:");
                    if ui
                        .add(egui::Slider::new(&mut pos.2, -5.0..=5.0).step_by(0.1))
                        .changed()
                    {
                        needs_render = true;
                    }
                });

                if needs_render {
                    surface.set_position(pos);
                }

                ui.separator();
                ui.label("Material:");

                let mut color = [
                    material.diffuse_color.0,
                    material.diffuse_color.1,
                    material.diffuse_color.2,
                ];

                if ui.color_edit_button_rgb(&mut color).changed() {
                    material.diffuse_color = Color(color[0], color[1], color[2]);
                    material.ambient_color = material.diffuse_color;
                    *surface.material_mut() = material;
                    needs_render = true;
                }
            }

            if needs_render && self.render_settings.auto_render {
                self.trigger_render();
            }
        }
    }

    /// Renders the light controls section
    fn render_light_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Scene Lights:");

        if ui.button("Add Light").clicked() {
            self.add_light();
        }

        if ui.button("Delete Selected Light").clicked() {
            self.delete_selected_light();
        }

        ui.separator();
        ui.label("Lights:");

        for (display_idx, light) in self.scene.light_sources.iter().enumerate() {
            let id = light.id();
            let is_selected = self.selection_state.selected_light_id == Some(id);
            let label = format!(
                "Light {} - Intensity: {:.2} (ID: {})",
                display_idx + 1,
                (light.intensity.0 + light.intensity.1 + light.intensity.2) / 3.0,
                id
            );

            if ui.selectable_label(is_selected, label).clicked() {
                self.selection_state.select_light(id);
            }
        }

        // Show selected light properties
        self.render_selected_light_properties(ui);
    }

    /// Renders the selected light properties section
    fn render_selected_light_properties(&mut self, ui: &mut egui::Ui) {
        if let Some(light_id) = self.selection_state.selected_light_id {
            if let Some(light) = self.scene.get_light_mut(light_id) {
                ui.separator();
                ui.label("Selected Light Properties:");

                let mut needs_render = false;

                ui.horizontal(|ui| {
                    ui.label("X:");
                    if ui
                        .add(egui::Slider::new(&mut light.position.0, -20.0..=20.0).step_by(0.1))
                        .changed()
                    {
                        needs_render = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Y:");
                    if ui
                        .add(egui::Slider::new(&mut light.position.1, -20.0..=20.0).step_by(0.1))
                        .changed()
                    {
                        needs_render = true;
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Z:");
                    if ui
                        .add(egui::Slider::new(&mut light.position.2, -20.0..=20.0).step_by(0.1))
                        .changed()
                    {
                        needs_render = true;
                    }
                });

                ui.separator();
                ui.label("Light Color & Intensity:");

                let mut color = [light.intensity.0, light.intensity.1, light.intensity.2];

                if ui.color_edit_button_rgb(&mut color).changed() {
                    light.intensity = Color(color[0], color[1], color[2]);
                    needs_render = true;
                }

                // Overall intensity slider
                let mut overall_intensity =
                    (light.intensity.0 + light.intensity.1 + light.intensity.2) / 3.0;
                ui.horizontal(|ui| {
                    ui.label("Brightness:");
                    if ui
                        .add(egui::Slider::new(&mut overall_intensity, 0.0..=1.0).step_by(0.01))
                        .changed()
                    {
                        let sum = light.intensity.0 + light.intensity.1 + light.intensity.2;
                        let ratio = if sum > 0.0 {
                            overall_intensity * 3.0 / sum
                        } else {
                            1.0
                        };
                        light.intensity = light.intensity * ratio;
                        needs_render = true;
                    }
                });

                if needs_render && self.render_settings.auto_render {
                    self.trigger_render();
                }
            }
        }
    }

    /// Renders the preview panel with the rendered image
    fn render_preview_panel(&self, ui: &mut egui::Ui, ctx: &egui::Context) {
        egui::ScrollArea::both().show(ui, |ui| match self.render_manager.preview_pixels.lock() {
            Ok(pixels) => {
                if pixels.is_empty() {
                    self.render_empty_preview_state(ui);
                    return;
                }

                if !self.is_pixel_buffer_ready(&pixels) {
                    self.render_preparing_state(ui);
                    return;
                }

                self.render_image_preview(ui, ctx, &pixels);
            }
            Err(_) => {
                ui.colored_label(
                    egui::Color32::RED,
                    "Error: Cannot access pixel buffer (mutex poisoned)",
                );
            }
        });
    }

    /// Checks if the pixel buffer is ready for display
    fn is_pixel_buffer_ready(&self, pixels: &[u8]) -> bool {
        let expected_size = (self.render_settings.resolution
            * self.render_settings.resolution
            * config::RGBA_CHANNELS) as usize;
        let is_zero_buffer = pixels.iter().all(|&byte| byte == 0);

        pixels.len() == expected_size && !is_zero_buffer
    }

    /// Renders the preparing state when buffer is being set up
    fn render_preparing_state(&self, ui: &mut egui::Ui) {
        if self.render_manager.pending_resolution.is_some() {
            ui.label("Preparing new resolution...");
        } else if self.render_manager.in_progress {
            ui.label("🔄 Rendering...");
        } else {
            ui.label("Click 'Render' to generate preview");
        }
    }

    /// Renders the state when no pixels are available
    fn render_empty_preview_state(&self, ui: &mut egui::Ui) {
        if self.render_manager.pending_resolution.is_some() {
            ui.label("Preparing new resolution...");
        } else {
            ui.label("Click 'Render' to generate preview");
        }
    }

    /// Renders the actual image preview with metadata
    fn render_image_preview(&self, ui: &mut egui::Ui, ctx: &egui::Context, pixels: &[u8]) {
        let color_image = if self.render_settings.resolution <= config::DOWNSAMPLING_THRESHOLD {
            // Use pixels directly for small resolutions
            egui::ColorImage::from_rgba_unmultiplied(
                [
                    self.render_settings.resolution as usize,
                    self.render_settings.resolution as usize,
                ],
                pixels,
            )
        } else {
            // Downsample for large resolutions
            let (display_pixels, half_res) = self.downsample_pixels(pixels);
            egui::ColorImage::from_rgba_unmultiplied(
                [half_res as usize, half_res as usize],
                &display_pixels,
            )
        };

        let texture = ctx.load_texture("preview", color_image, Default::default());
        let display_size = egui::Vec2::splat(config::DISPLAY_IMAGE_SIZE);
        ui.add(egui::Image::from_texture(&texture).fit_to_exact_size(display_size));

        self.render_image_metadata(ui);
    }

    /// Downsamples pixels for performance when resolution is high
    /// Returns the downsampled pixel data and the new resolution
    fn downsample_pixels(&self, pixels: &[u8]) -> (Vec<u8>, u32) {
        let half_res = self.render_settings.resolution / 2;
        let mut subsampled =
            Vec::with_capacity((half_res * half_res * config::RGBA_CHANNELS) as usize);
        let full_res = self.render_settings.resolution as usize;
        let channels = config::RGBA_CHANNELS as usize;

        for y in 0..half_res {
            for x in 0..half_res {
                let src_x = (x * 2) as usize;
                let src_y = (y * 2) as usize;
                let src_idx = (src_y * full_res + src_x) * channels;

                // Bounds check to prevent buffer overrun
                if src_idx + channels <= pixels.len() {
                    subsampled.extend_from_slice(&pixels[src_idx..src_idx + channels]);
                } else {
                    // Fill with black if we're out of bounds (shouldn't happen with correct sizing)
                    eprintln!("Warning: Downsampling bounds error at ({x}, {y}), index {src_idx}, buffer size {}", pixels.len());
                    subsampled.extend_from_slice(&[0, 0, 0, 255]);
                }
            }
        }

        (subsampled, half_res)
    }

    /// Renders metadata about the current image
    fn render_image_metadata(&self, ui: &mut egui::Ui) {
        ui.label(format!(
            "Rendered: {}x{}",
            self.render_settings.resolution, self.render_settings.resolution
        ));
        ui.label("Display Size: 800x800 (fixed)");

        if self.render_settings.resolution > config::DOWNSAMPLING_THRESHOLD {
            ui.label("(Display downsampled for performance)");
        }

        if let Some(pending) = self.render_manager.pending_resolution {
            ui.label(format!("Pending: {pending}x{pending}"));
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1400.0, 900.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Ray Tracer GUI",
        options,
        Box::new(|_cc| Ok(Box::new(RayTracerApp::new()))),
    )
}

impl Drop for RenderManager {
    fn drop(&mut self) {
        self.cancel_current_render();
    }
}
