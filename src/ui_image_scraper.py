import os, json, requests, io
from PIL import Image


class UIImageScraper:
    """A class for downloading and processing images based on configuration file."""
    
    def __init__(self, config_path="cm/ui_scraper_config.json", output_dir=None):
        """
        Initialize the UIImageScraper with configuration.
        
        Args:
            config_path (str): Path to the configuration JSON file
            output_dir (str): Optional override for output directory
        """
        self.config_path = config_path
        self.output_dir_override = output_dir
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def download_images_to_local(self):
        """Download and process images according to configuration. Returns a list of successfully downloaded file paths."""
        # Use override directory or config directory
        output_dir = self.output_dir_override or self.config.get('output_directory', 'images')
        os.makedirs(output_dir, exist_ok=True)

        image_data = self.config.get('image_data', [])
        headers = self.config.get('request_headers', {})
        transform_params = self.config.get('transform_parameters', {})
        image_quality = self.config.get('image_quality', 95)

        successful_files = []

        for item in image_data:
            url = item.get("url")
            filename = item.get("filename")
            filepath = os.path.join(output_dir, filename)

            try:
                response = requests.get(url, stream=True, headers=headers)

                if response.status_code == 200:
                    image = Image.open(io.BytesIO(response.content))

                    # Convert to grayscale if specified
                    if transform_params.get('convert_to_grayscale', False):
                        image = image.convert('L')

                    # Resize if enabled
                    resize_config = transform_params.get('resize', {})
                    if resize_config.get('enabled', False):
                        base_size = resize_config.get('base_size', 280)
                        aspect_ratio = resize_config.get('aspect_ratio', {})
                        width_mult = aspect_ratio.get('width_multiplier', 4)
                        height_mult = aspect_ratio.get('height_multiplier', 3)

                        dimensions = (base_size * width_mult, base_size * height_mult)
                        resampling = getattr(Image.Resampling, resize_config.get('resampling', 'LANCZOS'))
                        image = image.resize(dimensions, resampling)

                    # Save image
                    output_format = transform_params.get('output_format', 'JPEG')
                    if output_format.upper() == 'JPEG':
                        image.save(filepath, output_format, quality=image_quality)
                    else:
                        image.save(filepath, output_format)

                    print(f"✓ Downloaded and processed: {filepath}")
                    successful_files.append(filepath)

                else:
                    print(f"✗ Failed to fetch {filename}. Status code: {response.status_code}")

            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")

        return successful_files


if __name__ == "__main__":
    scraper = UIImageScraper()
    scraper.download_images_to_local()
    print("\nAll images processed!")