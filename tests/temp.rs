#[cfg(test)]
mod tests {
    use std::process::Command;
    use std::path::Path;
    use image::{GenericImageView, DynamicImage};
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_x86() {
        // Regenerate the image with compute_mosaic
        // Use the std::process::Command struct to run a command (in this case, "cargo")
        let output = Command::new("cargo")
            // Specify the arguments for the cargo command
            .args(&["run", "--release", "--", "--image", "assets/kit.jpeg", "--tiles", "assets/images", "--output", "out.png", "--tile-size" , "25" ])
            // Execute the command and capture its output
            .output()
            // Handle any errors that may occur during command execution
            .expect("Failed to execute command");
        // Assert that the command executed successfully (status code 0)
        assert!(output.status.success(), "Failed to execute compute_mosaic: {:?}", output);

        // Compare the generated image with the ground truth
        let generated_image_path = "out.png";
        let ground_truth_image_path = "assets/ground-truth-kit.png";

        let generated_image = image::open(generated_image_path).expect("Failed to open generated image");
        let ground_truth_image = image::open(ground_truth_image_path).expect("Failed to open ground truth image");

        assert_eq!(
            generated_image.dimensions(),
            ground_truth_image.dimensions(),
            "Dimensions of the generated image do not match the ground truth"
        );

        

        // Clean up the generated file
        if Path::new(generated_image_path).exists() {
            std::fs::remove_file(generated_image_path).unwrap();
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64() {
        // Regenerate the image with compute_mosaic
        let output = Command::new("cargo")
            .args(&["run", "--release", "--", "--image", "assets/kit.jpeg", "--tiles", "assets/images", "--output", "out.png", "--tile-size" , "25" ])
            .output()
            .expect("Failed to execute command");

        assert!(output.status.success(), "Failed to execute compute_mosaic: {:?}", output);

        // Compare the generated image with the ground truth
        let generated_image_path = "out.png";
        let ground_truth_image_path = "assets/ground-truth-kit.png";

        let generated_image = image::open(generated_image_path).expect("Failed to open generated image");
        let ground_truth_image = image::open(ground_truth_image_path).expect("Failed to open ground truth image");

        assert_eq!(
            generated_image.dimensions(),
            ground_truth_image.dimensions(),
            "Dimensions of the generated image do not match the ground truth"
        );

       

        // Clean up the generated file
        if Path::new(generated_image_path).exists() {
            std::fs::remove_file(generated_image_path).unwrap();
        }
    }


    
    #[test]
    fn test_compare_with_ground_truth() {
        // Regenerate the image with compute_mosaic
        let output = Command::new("cargo")
            .args(&["run", "--release", "--", "--image", "assets/kit.jpeg", "--tiles", "assets/images", "--output", "out.png", "--tile-size" , "25" ])
            .output()
            .expect("Failed to execute command");

        assert!(output.status.success(), "Failed to execute compute_mosaic: {:?}", output);

        // Compare the generated image with the ground truth
        let generated_image_path = "out.png";
        let ground_truth_image_path = "assets/ground-truth-kit.png";

        let generated_image = image::open(generated_image_path).expect("Failed to open generated image");
        let ground_truth_image = image::open(ground_truth_image_path).expect("Failed to open ground truth image");

        assert_eq!(
            generated_image.dimensions(),
            ground_truth_image.dimensions(),
            "Dimensions of the generated image do not match the ground truth"
        );

        
        // Clean up the generated file
        if Path::new(generated_image_path).exists() {
            std::fs::remove_file(generated_image_path).unwrap();
        }
    }
}
