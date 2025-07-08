use ou_graphics::debug::set_debug_mode;
use ou_graphics::debug_println;
use ou_graphics::primitives::*;
use ou_graphics::render::*;
use ou_graphics::types::image::*;
use ou_graphics::types::linalg::*;
use piston_window::EventLoop;

const WIDTH: u32 = 720;
const HEIGHT: u32 = 720;
const CLEAR_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0]; // White background

fn my_scene() -> Scene {
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

    let main_light = Light::white(Vector3(16.0, -16.0, 16.0) * 1.2, 0.15);
    let fill_light = Light::new(Vector3(-10.0, 10.0, 20.0), Color::CYAN * 0.05);

    Scene::default()
        .with_background(Color::WHITE)
        .with_ambient_light(Color::WHITE * 0.3)
        .with_surface(Box::new(green_sphere))
        .with_surface(Box::new(red_sphere))
        .with_surface(Box::new(blue_sphere))
        .with_light(main_light)
        .with_light(fill_light)
}

fn my_camera() -> Camera {
    Camera::new(
        Vector3(5.0, 0.0, 0.0),
        Vector3::ORIGIN, // Point to green sphere
        Vector3::Z,
        80.0,
        (WIDTH as usize, HEIGHT as usize),
    )
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let debug_enabled = args.iter().any(|arg| arg == "--debug");
    set_debug_mode(debug_enabled);
    debug_println!("Debug mode is enabled");

    let scene = my_scene();
    let camera = my_camera();
    let frame = render(&camera, &scene);

    if args.iter().any(|arg| arg == "--text") {
        print_image_as_text(&frame);
        return;
    }

    let mut frame_buffer = image::ImageBuffer::new(WIDTH, HEIGHT);
    for i in 0..frame.cols() {
        for j in 0..frame.rows() {
            let color = frame[[i, j]].0;
            let (r, g, b) = color.to_rgb8();
            let pixel = image::Rgba([r, g, b, 255]);
            frame_buffer.put_pixel(i as u32, j as u32, pixel);
        }
    }

    let mut window: piston_window::PistonWindow =
        piston_window::WindowSettings::new("ou-graphics", [WIDTH, HEIGHT])
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
