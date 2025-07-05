extern crate image;
extern crate piston_window;

use piston_window::EventLoop;

use ou_graphics::primitives::*;
use ou_graphics::render::*;
use ou_graphics::types::*;

const WIDTH: u32 = 720;
const HEIGHT: u32 = 720;

fn my_sphere() -> Sphere {
    let green = Color(
        rgb8_to_colorchannel(64),
        rgb8_to_colorchannel(255),
        rgb8_to_colorchannel(64),
    );
    let grey = Color(
        rgb8_to_colorchannel(64),
        rgb8_to_colorchannel(64),
        rgb8_to_colorchannel(64),
    );
    let material = Material {
        shininess: 1000.0,
        diffuse_color: green,
        specular_color: grey * 4.0,
        ambient_color: green,
        mirror_color: grey,
    };
    Sphere {
        center: Vector3(0.0, 0.0, 0.0),
        radius: 14.0,
        material,
    }
}

fn my_light() -> Light {
    let white = Color(
        rgb8_to_colorchannel(255),
        rgb8_to_colorchannel(255),
        rgb8_to_colorchannel(255),
    );
    Light {
        position: Vector3(16.0, -16.0, 16.0) * 1.2,
        //position: Vector3(8.0, 0.0, 8.0),
        intensity: white * 0.2,
    }
}

fn my_scene() -> Scene {
    let sphere = my_sphere();
    //let floor = my_floor();
    let light = my_light();
    let dark_grey = Color(
        rgb8_to_colorchannel(16),
        rgb8_to_colorchannel(16),
        rgb8_to_colorchannel(16),
    );
    //let almost_black = dark_grey * 0.25;
    Scene {
        background_color: dark_grey,
        surfaces: vec![Box::new(sphere)],
        //surfaces: vec![],
        light_sources: vec![light],
        //light_sources: vec![],
        ambient_light_intensity: dark_grey * 1.0,
    }
}

fn my_camera() -> Camera {
    let basis = Basis3(
        Vector3(0.0, 1.0, 0.0),
        Vector3(0.0, 0.0, 1.0),
        Vector3(1.0, 0.0, 0.0),
    );
    Camera {
        position: Vector3(14.5, 0.0, 0.0),
        basis,
        resolution: (WIDTH as usize, HEIGHT as usize),
        focal_length: 32.0,
    }
}

fn print_image_as_text(image: &Image) {
    for i in 0..image.cols() {
        for j in 0..image.rows() {
            let Color(r, g, b) = image[[i, j]].0;
            print!("{}", format!("({:3},{:3},{:3})", r, g, b));
        }
        println!("");
        println!("");
        println!("");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
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
            let Color(r, g, b) = frame[[i, j]].0;
            let pixel = image::Rgba([(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]);
            frame_buffer.put_pixel(i as u32, j as u32, pixel);
        }
    }

    let mut window: piston_window::PistonWindow =
        piston_window::WindowSettings::new("Raytracer", [WIDTH, HEIGHT])
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
            piston_window::clear([1.0; 4], g);
            piston_window::image(&tex, c.transform, g)
        });
    }
}
