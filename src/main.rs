use ou_graphics::debug::{set_debug_mode, set_verbose_mode};
use ou_graphics::primitives::*;
use ou_graphics::render::*;
use ou_graphics::types::image::*;
use ou_graphics::types::linalg::*;
use ou_graphics::{debug_println, verbose_println};
use std::time::Instant;

fn spheres_scene() -> Scene {
    let green_sphere = Sphere::new(
        Vector3::ORIGIN,
        1.0,
        Material {
            shininess: 500.0,
            diffuse_color: Color::GREEN,
            specular_color: Color::WHITE * 0.5,
            ambient_color: Color::GREEN,
            mirror_color: Color::WHITE * 0.25,
        },
    );

    let red_sphere = Sphere::new(
        Vector3(2.0, -0.5, 1.5),
        0.6,
        Material {
            shininess: 300.0,
            diffuse_color: Color::RED,
            specular_color: Color::WHITE * 0.6,
            ambient_color: Color::RED,
            mirror_color: Color::WHITE * 0.2,
        },
    );

    let blue_sphere = Sphere::new(
        Vector3(2.5, -0.75, -0.75),
        0.6,
        Material {
            shininess: 200.0,
            diffuse_color: Color::BLUE,
            specular_color: Color::WHITE * 0.4,
            ambient_color: Color::BLUE,
            mirror_color: Color::WHITE * 0.3,
        },
    );

    let main_light = Light::white(Vector3(0.0, 0.0, 100.0), 0.75);
    let fill_light = Light::new(Vector3(10.0, 10.0, 0.0), Color::CYAN * 0.15);

    Scene::default()
        .with_background(Color::WHITE)
        .with_ambient_light(Color::WHITE * 0.3)
        .with_surface(Box::new(green_sphere))
        .with_surface(Box::new(red_sphere))
        .with_surface(Box::new(blue_sphere))
        .with_light(main_light)
        .with_light(fill_light)
}

fn polygon_scene() -> Scene {
    let vertices = vec![
        Vector3(0.0, 1.0, 0.0),
        Vector3(0.0, -1.0, -1.0),
        Vector3(0.0, -1.0, 1.0),
    ];
    let material = Material {
        shininess: 500.0,
        diffuse_color: Color::GREEN,
        specular_color: Color::WHITE * 0.5,
        ambient_color: Color::GREEN,
        mirror_color: Color::WHITE * 0.25,
    };
    let polygon = Polygon::new(vertices, material);
    let light = Light::white(Vector3(100.0, 0.0, 0.0), 0.8);
    Scene::default()
        .with_background(Color::WHITE)
        .with_ambient_light(Color::WHITE * 0.2)
        .with_surface(Box::new(polygon))
        .with_light(light)
}

fn mirror_ball_scene() -> Scene {
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

fn print_image_as_text(image: &Image) {
    for i in 0..image.cols() {
        for j in 0..image.rows() {
            let Color(r, g, b) = image[[i, j]].0;
            print!("({r:3},{g:3},{b:3})");
        }
        println!("\n\n");
    }
}

struct Options {
    debug_enabled: bool,
    verbose_enabled: bool,
    render_width: u32,
    render_height: u32,
    display_width: u32,
    display_height: u32,
    camera_distance: f32,
    fov: f32,
    output_path: Option<String>,
    text_mode: bool,
    no_window: bool,
    max_depth: Option<usize>,
    scene: String,
    single_pixel: Option<(u32, u32)>,
    threads: usize,
    samples_per_pixel: usize,
}

impl Options {
    fn from_args(args: &[String]) -> Self {
        let mut render_width = 720;
        let mut render_height = 720;
        let mut display_width = 720;
        let mut display_height = 720;
        let mut camera_distance = 5.0;
        let mut fov = 80.0;
        let mut output_path = None;
        let mut debug_enabled = false;
        let mut verbose_enabled = false;
        let mut text_mode = false;
        let mut no_window = false;
        let mut max_depth = None;
        let mut scene = "mirror_ball".to_string();
        let mut single_pixel = None;
        let mut threads = 8;
        let mut samples_per_pixel = 32;
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--debug" => debug_enabled = true,
                "--verbose" => verbose_enabled = true,
                "--text" => text_mode = true,
                "--no-window" => no_window = true,
                "--resolution" => {
                    if i + 1 < args.len() {
                        if let Some((w, h)) = args[i + 1].split_once('x') {
                            if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                                render_width = w;
                                render_height = h;
                            }
                        }
                        i += 1;
                    }
                }
                "--display" => {
                    if i + 1 < args.len() {
                        if let Some((w, h)) = args[i + 1].split_once('x') {
                            if let (Ok(w), Ok(h)) = (w.parse::<u32>(), h.parse::<u32>()) {
                                display_width = w;
                                display_height = h;
                            }
                        }
                        i += 1;
                    }
                }
                "--camera-distance" => {
                    if i + 1 < args.len() {
                        if let Ok(dist) = args[i + 1].parse::<f32>() {
                            camera_distance = dist;
                        }
                        i += 1;
                    }
                }
                "--fov" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<f32>() {
                            fov = val;
                        }
                        i += 1;
                    }
                }
                "--output" => {
                    if i + 1 < args.len() {
                        output_path = Some(args[i + 1].clone());
                        i += 1;
                    }
                }
                "--max-depth" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<usize>() {
                            max_depth = Some(val);
                        }
                        i += 1;
                    }
                }
                "--scene" => {
                    if i + 1 < args.len() {
                        scene = args[i + 1].clone();
                        i += 1;
                    }
                }
                "--single-pixel" => {
                    if i + 1 < args.len() {
                        if let Some((x, y)) = args[i + 1].split_once(',') {
                            if let (Ok(x), Ok(y)) = (x.parse::<u32>(), y.parse::<u32>()) {
                                single_pixel = Some((x, y));
                            }
                        }
                        i += 1;
                    }
                }
                "--threads" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<usize>() {
                            threads = val.max(1);
                        }
                        i += 1;
                    }
                }
                "--samples" => {
                    if i + 1 < args.len() {
                        if let Ok(val) = args[i + 1].parse::<usize>() {
                            samples_per_pixel = val.max(1);
                        }
                        i += 1;
                    }
                }
                _ => {}
            }
            i += 1;
        }
        Self {
            debug_enabled,
            verbose_enabled,
            render_width,
            render_height,
            display_width,
            display_height,
            camera_distance,
            fov,
            output_path,
            text_mode,
            no_window,
            max_depth,
            scene,
            single_pixel,
            threads,
            samples_per_pixel,
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let options = Options::from_args(&args);
    set_debug_mode(options.debug_enabled);
    set_verbose_mode(options.verbose_enabled);
    debug_println!("Debug mode is enabled");
    verbose_println!("Verbose mode is enabled");

    let render_aspect = options.render_width as f32 / options.render_height as f32;
    let display_aspect = options.display_width as f32 / options.display_height as f32;
    if (render_aspect - display_aspect).abs() > 0.01 {
        eprintln!(
            "Warning: Render aspect ratio ({render_aspect:.3}) and display aspect ratio ({display_aspect:.3}) differ. Image may appear stretched or squashed."
        );
    }

    let scene = match options.scene.as_str() {
        "polygon" => polygon_scene(),
        "spheres" => spheres_scene(),
        "mirror_ball" => mirror_ball_scene(),
        other => {
            eprintln!("Error: unrecognized scene '{other}'.");
            std::process::exit(1);
        }
    };
    let camera = Camera::new(
        Vector3(options.camera_distance, 0.0, 0.0),
        Vector3::ORIGIN,
        Vector3::Z,
        options.fov,
        (
            options.render_width as usize,
            options.render_height as usize,
        ),
    );
    let max_depth = options.max_depth.unwrap_or(5);
    let start = Instant::now();
    let config = RenderConfig::new(
        camera,
        scene,
        max_depth,
        options.threads,
        options.samples_per_pixel,
    );
    let frame = render(config);
    let duration = start.elapsed();
    debug_println!("Render took {:.2?}", duration);

    if options.text_mode {
        print_image_as_text(&frame);
        return;
    }

    let mut frame_buffer = image::ImageBuffer::new(options.render_width, options.render_height);
    for i in 0..frame.cols() {
        for j in 0..frame.rows() {
            let color = frame[[i, j]].0;
            let (r, g, b) = color.to_srgb8();
            let pixel = image::Rgba([r, g, b, 255]);
            frame_buffer.put_pixel(i as u32, j as u32, pixel);
        }
    }

    // Single pixel mode: fill the frame_buffer with the color of the selected pixel
    if let Some((px, py)) = options.single_pixel {
        if px < options.render_width && py < options.render_height {
            let color = frame[[px as usize, py as usize]].0;
            let (r, g, b) = color.to_srgb8();
            let pixel = image::Rgba([r, g, b, 255]);
            for i in 0..options.render_width {
                for j in 0..options.render_height {
                    frame_buffer.put_pixel(i, j, pixel);
                }
            }
        } else {
            eprintln!("Warning: --single-pixel coordinates out of bounds");
        }
    }

    if let Some(path) = &options.output_path {
        frame_buffer.save(path).expect("Failed to save image");
        println!("Image saved to {path}");
        return;
    }

    if options.no_window {
        return;
    }

    let display_buffer = if options.display_width != options.render_width
        || options.display_height != options.render_height
    {
        image::imageops::resize(
            &frame_buffer,
            options.display_width,
            options.display_height,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        frame_buffer
    };

    // Save to a temporary file and open with default viewer
    let tmp_path = std::env::temp_dir().join("ou-graphics-preview.png");
    display_buffer
        .save(&tmp_path)
        .expect("Failed to save temp image");
    open::that(&tmp_path).expect("Failed to open image in default viewer");
}
