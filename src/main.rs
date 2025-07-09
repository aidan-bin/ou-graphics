use ou_graphics::debug::{set_debug_mode, set_verbose_mode};
use ou_graphics::primitives::*;
use ou_graphics::render::*;
use ou_graphics::types::image::*;
use ou_graphics::types::linalg::*;
use ou_graphics::{debug_println, verbose_println};
use piston_window::EventLoop;

const CLEAR_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0]; // White background

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
    width: u32,
    height: u32,
    camera_distance: f32,
    fov: f32,
    ambient: Option<(f32, f32, f32)>,
    output_path: Option<String>,
    text_mode: bool,
    no_window: bool,
    max_depth: Option<usize>,
    scene: String,
}

impl Options {
    fn from_args(args: &[String]) -> Self {
        let mut width = 720;
        let mut height = 720;
        let mut camera_distance = 5.0;
        let mut fov = 80.0;
        let mut ambient = None;
        let mut output_path = None;
        let mut debug_enabled = false;
        let mut verbose_enabled = false;
        let mut text_mode = false;
        let mut no_window = false;
        let mut max_depth = None;
        let mut scene = "spheres".to_string();
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
                                width = w;
                                height = h;
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
                "--ambient" => {
                    if i + 1 < args.len() {
                        let parts: Vec<_> = args[i + 1].split(',').collect();
                        if parts.len() == 3 {
                            if let (Ok(r), Ok(g), Ok(b)) = (
                                parts[0].parse::<f32>(),
                                parts[1].parse::<f32>(),
                                parts[2].parse::<f32>(),
                            ) {
                                ambient = Some((r, g, b));
                            }
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
                _ => {}
            }
            i += 1;
        }
        Self {
            debug_enabled,
            verbose_enabled,
            width,
            height,
            camera_distance,
            fov,
            ambient,
            output_path,
            text_mode,
            no_window,
            max_depth,
            scene,
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

    let scene = {
        let mut scene = match options.scene.as_str() {
            "polygon" => polygon_scene(),
            "spheres" => spheres_scene(),
            other => {
                eprintln!("Error: unrecognized scene '{other}'.");
                std::process::exit(1);
            }
        };
        if let Some((r, g, b)) = options.ambient {
            scene = scene.with_ambient_light(Color(r, g, b));
        }
        scene
    };
    let camera = Camera::new(
        Vector3(options.camera_distance, 0.0, 0.0),
        Vector3::ORIGIN,
        Vector3::Z,
        options.fov,
        (options.width as usize, options.height as usize),
    );
    let max_depth = options.max_depth.unwrap_or(5);
    let frame = render(&camera, &scene, max_depth);

    if options.text_mode {
        print_image_as_text(&frame);
        return;
    }

    let mut frame_buffer = image::ImageBuffer::new(options.width, options.height);
    for i in 0..frame.cols() {
        for j in 0..frame.rows() {
            let color = frame[[i, j]].0;
            let (r, g, b) = color.to_rgb8();
            let pixel = image::Rgba([r, g, b, 255]);
            frame_buffer.put_pixel(i as u32, j as u32, pixel);
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

    let mut window: piston_window::PistonWindow =
        piston_window::WindowSettings::new("ou-graphics", [options.width, options.height])
            .exit_on_esc(true)
            .build()
            .unwrap_or_else(|_e| panic!("Could not create window!"));

    let tex = piston_window::Texture::from_image(
        &mut window.create_texture_context(),
        &frame_buffer,
        &piston_window::TextureSettings::new(),
    )
    .unwrap();

    window.set_lazy(true);

    while let Some(e) = window.next() {
        window.draw_2d(&e, |c, g, _| {
            piston_window::clear(CLEAR_COLOR, g);
            piston_window::image(&tex, c.transform, g)
        });
    }
}
