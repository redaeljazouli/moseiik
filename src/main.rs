use clap::Parser;
use image::{
    imageops::{resize, FilterType::Nearest},
    io::Reader as ImageReader,
    GenericImage, GenericImageView, RgbImage,
};
use std::time::Instant;
use std::{
    error::Error,
    fs,
    ops::Deref,
    sync::{Arc, Mutex},
};
use threadpool::ThreadPool;
use threadpool_scope::scope_with;

#[derive(Debug, Parser)]
struct Size {
    width: u32,
    height: u32,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Options {
    /// Location of the target image
    #[arg(short, long)]
    image: String,

    /// Saved result location
    #[arg(short, long, default_value_t=String::from("out.png"))]
    output: String,

    /// Location of the tiles
    #[arg(short, long)]
    tiles: String,

    /// Scaling factor of the image
    #[arg(long, default_value_t = 1)]
    scaling: u32,

    /// Size of the tiles
    #[arg(long, default_value_t = 5)]
    tile_size: u32,

    /// Remove used tile
    #[arg(short, long)]
    remove_used: bool,

    #[arg(short, long)]
    verbose: bool,

    /// Use SIMD when available
    #[arg(short, long)]
    simd: bool,

    /// Specify number of threads to use, leave blank for default
    #[arg(short, long, default_value_t = 1)]
    num_thread: usize,
}

fn count_available_tiles(images_folder: &str) -> i32 {
    match fs::read_dir(images_folder) {
        Ok(t) => return t.count() as i32,
        Err(_) => return -1,
    };
}

fn prepare_tiles(images_folder: &str, tile_size: &Size, verbose: bool) -> Result<Vec<RgbImage>, Box<dyn Error>> {
    let image_paths = fs::read_dir(images_folder)?;
    let tiles = Arc::new(Mutex::new(Vec::new()));
    let now = Instant::now();
    let pool = ThreadPool::new(num_cpus::get());
    let tile_width = tile_size.width;
    let tile_height = tile_size.height;

    for image_path in image_paths {
        let tiles = Arc::clone(&tiles);
        pool.execute(move || {
            let tile_result =
                || -> Result<RgbImage, Box<dyn Error>> { Ok(ImageReader::open(image_path?.path())?.decode()?.into_rgb8()) };

            let tile = match tile_result() {
                Ok(t) => t,
                Err(_) => return,
            };

            let tile = resize(&tile, tile_width, tile_height, Nearest);
            tiles.lock().unwrap().push(tile)
        });
    }
    pool.join();

    println!(
        "\n{} elements in {} seconds",
        tiles.lock().unwrap().len(),
        now.elapsed().as_millis() as f32 / 1000.0
    );

    if verbose {
        println!("");
    }
    let res = tiles.lock().unwrap().deref().to_owned();
    return Ok(res);
}

fn l1_generic(im1: &RgbImage, im2: &RgbImage) -> i32 {
    im1.iter()
        .zip(im2.iter())
        .fold(0, |res, (a, b)| res + i32::abs((*a as i32) - (*b as i32)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn l1_x86_sse2(im1: &RgbImage, im2: &RgbImage) -> i32 {
    // Only works if data is 16 bytes-aligned, which should be the case.
    // In case of crash due to unaligned data, swap _mm__si128 for _mm_loadu_si128.
    use std::arch::x86_64::{
        __m128i,
        _mm_extract_epi16, //SSE2
        _mm_load_si128,    //SSE2
        _mm_sad_epu8,      //SSE2
    };

    let stride = 128 / 8;

    let tile_size = im1.width() * im1.height();
    let nb_sub_pixel = tile_size * 3;

    let im1 = im1.as_raw();
    let im2 = im2.as_raw();

    let mut result: i32 = 0;

    for i in (0..nb_sub_pixel - stride).step_by(stride as usize) {
        // Get pointer to data
        let p_im1: *const __m128i = std::mem::transmute::<*const u8, *const __m128i>(std::ptr::addr_of!(im1[i as usize]));
        let p_im2: *const __m128i = std::mem::transmute::<*const u8, *const __m128i>(std::ptr::addr_of!(im2[i as usize]));

        // Load data to xmm
        let xmm_p1 = _mm_load_si128(p_im1);
        let xmm_p2 = _mm_load_si128(p_im2);

        // Do abs(a-b) and horizontal add, results are stored in lower 16 bits of each 64 bits groups
        let xmm_sub_abs = _mm_sad_epu8(xmm_p1, xmm_p2);

        let res_0 = _mm_extract_epi16(xmm_sub_abs, 0);
        let res_1 = _mm_extract_epi16(xmm_sub_abs, 4);

        result += res_0 + res_1; // + res_2 + res_3;
    }

    // now do the remainder manually
    let remainder = nb_sub_pixel % stride as u32;
    for i in nb_sub_pixel - remainder..nb_sub_pixel {
        let p1: u8 = im1[i as usize];
        let p2: u8 = im2[i as usize];

        result += i32::abs((p1 as i32) - (p2 as i32));
    }

    return result;
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_neon(im1: &RgbImage, im2: &RgbImage) -> i32 {
    use std::arch::aarch64::uint8x16_t;
    use std::arch::aarch64::vabdq_u8; // Absolute subtract
    use std::arch::aarch64::vaddlvq_u8; // horizontal add
    use std::arch::aarch64::vld1q_u8; // Load instruction

    let stride = 128 / 8;

    let tile_size = im1.width() * im1.height();
    let nb_sub_pixel = tile_size * 3;

    let im1 = im1.as_raw();
    let im2 = im2.as_raw();

    let mut result: i32 = 0;

    for i in (0..nb_sub_pixel - stride).step_by(stride as usize) {
        // get pointer to data
        let p_im1: *const u8 = std::ptr::addr_of!(im1[i as usize]);
        let p_im2: *const u8 = std::ptr::addr_of!(im2[i as usize]);

        // load data to xmm
        let xmm1: uint8x16_t = vld1q_u8(p_im1);
        let xmm2: uint8x16_t = vld1q_u8(p_im2);

        // get absolute difference
        let xmm_abs_diff: uint8x16_t = vabdq_u8(xmm1, xmm2);

        // reduce with horizontal add
        result += vaddlvq_u8(xmm_abs_diff) as i32;
    }

    // now do the remainder manually
    let remainder = nb_sub_pixel % stride as u32;
    for i in nb_sub_pixel - remainder..nb_sub_pixel {
        let p1: u8 = im1[i as usize];
        let p2: u8 = im2[i as usize];

        result += i32::abs((p1 as i32) - (p2 as i32));
    }

    return result;
}

unsafe fn get_optimal_l1(simd_flag: bool, verbose: bool) -> unsafe fn(&RgbImage, &RgbImage) -> i32 {
    static mut FN_POINTER: unsafe fn(&RgbImage, &RgbImage) -> i32 = l1_generic;

    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        if simd_flag {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("sse2") {
                    if verbose {
                        println!("{}[2K\rUsing SSE2 SIMD.", 27 as char);
                    }
                    FN_POINTER = l1_x86_sse2;
                } else {
                    if verbose {
                        println!("{}[2K\rNot using SIMD.", 27 as char);
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::is_aarch64_feature_detected;
                if is_aarch64_feature_detected!("neon") {
                    if verbose {
                        println!("{}[2K\rUsing NEON SIMD.", 27 as char);
                    }
                    FN_POINTER = l1_neon;
                } else {
                    if verbose {
                        println!("{}[2K\rNot using SIMD.", 27 as char);
                    }
                }
            }
        }
    });

    return FN_POINTER;
}

fn l1(im1: &RgbImage, im2: &RgbImage, simd_flag: bool, verbose: bool) -> i32 {
    return unsafe { get_optimal_l1(simd_flag, verbose)(im1, im2) };
}

fn prepare_target(image_path: &str, scale: u32, tile_size: &Size) -> Result<RgbImage, Box<dyn Error>> {
    let target = ImageReader::open(image_path)?.decode()?.into_rgb8();
    let width = target.width();
    let height = target.height();
    let target = target
        .view(0, 0, width - width % tile_size.width, height - height % tile_size.height)
        .to_image();
    Ok(resize(&target, target.width() * scale, target.height() * scale, Nearest))
}

fn find_best_tile(target: &RgbImage, tiles: &Vec<RgbImage>, simd: bool, verbose: bool) -> usize {
    let mut index_best_tile = 0;
    let mut min_error = i32::MAX;
    for (i, tile) in tiles.iter().enumerate() {
        let error = l1(tile, &target, simd, verbose);
        if error < min_error {
            min_error = error;
            index_best_tile = i;
        }
    }
    return index_best_tile;
}

fn compute_mosaic(args: Options) {
    let tile_size = Size {
        width: args.tile_size,
        height: args.tile_size,
    };

    let (target_size, target) = match prepare_target(&args.image, args.scaling, &tile_size) {
        Ok(t) => (
            Size {
                width: t.width(),
                height: t.height(),
            },
            Arc::new(Mutex::new(t)),
        ),
        Err(e) => panic!("Error opening {}. {}", args.image, e),
    };

    let nb_available_tiles = count_available_tiles(&args.tiles);
    let nb_required_tiles: i32 = ((target_size.width / tile_size.width) * (target_size.height / tile_size.height)) as i32;
    if args.remove_used && nb_required_tiles > nb_available_tiles {
        panic!("{} tiles required, found {}.", nb_required_tiles, nb_available_tiles)
    }

    let tiles = &prepare_tiles(&args.tiles, &tile_size, args.verbose).unwrap();
    if args.verbose {
        println!("w: {}, h: {}", target_size.width, target_size.height);
    }

    let now = Instant::now();
    let pool = ThreadPool::new(args.num_thread);
    scope_with(&pool, |scope| {
        for w in 0..target_size.width / tile_size.width {
            let target = Arc::clone(&target);
            scope.execute(move || {
                for h in 0..target_size.height / tile_size.height {
                    if args.verbose {
                        print!(
                            "\rBuild image: {} / {} : {} / {}",
                            w + 1,
                            target_size.width / tile_size.width,
                            h + 1,
                            target_size.height / tile_size.height
                        );
                    }

                    // Crop the tile
                    let target_tile = &(target
                        .lock()
                        .unwrap()
                        .view(tile_size.width * w, tile_size.height * h, tile_size.width, tile_size.height)
                        .to_image());

                    let index_best_tile = find_best_tile(&target_tile, &tiles, args.simd, args.verbose);

                    target
                        .lock()
                        .unwrap()
                        .copy_from(&tiles[index_best_tile], w * tile_size.width, h * tile_size.height)
                        .unwrap();
                }
            });
        }
    });
    println!("\n{} seconds", now.elapsed().as_millis() as f32 / 1000.0);
    target.lock().unwrap().save(args.output).unwrap();
}

fn main() {
    let args = Options::parse();
    compute_mosaic(args);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;
    //on s'en servira de cette fonction dans les tests pour loader les images
    fn load_test_image(path: &str) -> RgbImage {
        ImageReader::open(path).unwrap().decode().unwrap().into_rgb8()
    }

    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn unit_test_x86() {
        
        let im1 = load_test_image("assets/tiles-small/tile-3.png");
        let im2 = load_test_image("assets/tiles-small/tile-3.png");

        // On calcule la distance L1 entre les deux memes images
        let distance = unsafe { l1_x86_sse2(&im1, &im2) };

        
        let expected_distance = 0;

        // on vérifie que la distance calculée est égale à la valeur attendue
        assert_eq!(distance, expected_distance, "Distance L1 does not match expected value");
    }

    #[test]
    fn test_prepare_target() {
    //Chemin vers l'image de test et paramètres de la tuile
    let test_image_path = "assets/kit.jpeg";
    let scale = 2;
    let tile_size = Size { width: 25, height: 25 };
    // Appel de la fonction `prepare_target` et vérification si le résultat est OK (pas d'erreur)
    let prepared_image_result = prepare_target(test_image_path, scale, &tile_size);
    assert!(prepared_image_result.is_ok());
    // Déballage du résultat pour obtenir l'image préparée
    let prepared_image = prepared_image_result.unwrap();

    // On calcule d'abord la largeur et la hauteur après le rognage
    let cropped_width = 1920 - 1920 % tile_size.width;
    let cropped_height = 1080 - 1080 % tile_size.height;

    // ensuite on applique le facteur d'échelle
    let expected_width = cropped_width * scale;
    let expected_height = cropped_height * scale;
    //verification des dimensions de l'image préparée correspondant aux dimensions attendues
    assert_eq!(prepared_image.width(), expected_width as u32, "image width does not match expected");
    assert_eq!(prepared_image.height(), expected_height as u32, "image height does not match expected");
}
    #[test]
    fn test_prepare_tiles() {
        //on défini le path de nos tiles
        let images_folder = "assets/tiles-small";
        let tile_size = Size { width: 50, height: 50 };
        let verbose = false;

        match prepare_tiles(images_folder, &tile_size, verbose) {
            Ok(tiles) => {
                // Vérification que le nombre d'images chargées correspond au nombre d'images dans le dossier
                let expected_num_images = std::fs::read_dir(images_folder).unwrap().count();
                assert_eq!(tiles.len(), expected_num_images, "Number of tiles does not match");

                // on vérifie que chaque tuile est redimensionnée correctement
                for tile in tiles {
                    assert_eq!(tile.width(), tile_size.width, "Tile width does not match");
                    assert_eq!(tile.height(), tile_size.height, "Tile height does not match");
                }
            },
            Err(e) => panic!("Failed to prepare tiles: {}", e),
        }
    }//on vérifie que toutes les images du dossier sont chargées et redimensionnées correctement selon la taille de tuile que nous avons spécifiée dans le test.


    // Les tests placeholders pour AArch64 et generic
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn unit_test_aarch64() {
        //c'est le meme principe de la distance L1 entre deux images
        let im1 = load_test_image("assets/tiles-small/tile-3.png");
        let im2 = load_test_image("assets/tiles-small/tile-3.png");

        // on calcule la distance L1 entre les deux images
        let distance = unsafe { l1_neon(&im1, &im2) };

        // normallement pour deux memes image on aurra 0
        let expected_distance = 0;

        // Vérification des résultats
        assert_eq!(distance, expected_distance, "distance L1 does not match expected value");
    }

    #[test]
    fn unit_test_generic() {
        //test de distance l1 generic
        let im1 = load_test_image("assets/tiles-small/tile-3.png");
        let im2 = load_test_image("assets/tiles-small/tile-3.png");
        let result = l1_generic(&im1, &im2);
        assert_eq!(result, 0);
    }
}
