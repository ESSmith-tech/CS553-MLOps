
import os, pytest, tempfile, shutil, json
from src.ui_image_scraper import UIImageScraper

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'cm', 'ui_scraper_config.json')

def load_image_data_count(config_path):
	with open(config_path, 'r', encoding='utf-8') as f:
		config = json.load(f)
	return len(config.get('image_data', []))

def test_download_images_to_temp_dir():
	temp_dir = tempfile.mkdtemp()
	try:
		scraper = UIImageScraper(config_path=CONFIG_PATH, output_dir=temp_dir)
		scraper.download_images_to_local()
		files = [f for f in os.listdir(temp_dir) if os.path.isfile(os.path.join(temp_dir, f))]
		expected_count = load_image_data_count(CONFIG_PATH)
		assert len(files) == expected_count, f"Expected {expected_count} images, found {len(files)}"
	finally:
		shutil.rmtree(temp_dir)
