from openai import OpenAI
import time
from tqdm import tqdm
import argparse
import os
import re
from typing import List, Dict, Tuple

def get_suburban_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)  # Go from preprocess folder to SubUrban root

class GenerationTemplate:
    def __init__(self, template):
        self.template = template

    def fill(self, city='', region=''):
        return self.template.replace('[CITY]', city).replace('[REGION]', region)

# Multi-city configuration - keys match the exact input format you'll use
CITY_CONFIGS = {
    "Beijing": {
        "name": "Beijing",
        "name_chn": "北京市",
        "districts": ["东城区", "西城区", "朝阳区", "海淀区", "丰台区", "石景山区", "通州区", 
                     "昌平区", "大兴区", "顺义区", "房山区", "门头沟区", "平谷区", "怀柔区", "密云区", "延庆区"],
        "language": "chinese",
        "dataset": "Gaode",
        "expertise": "Beijing geography expert",
        "use_chinese_prompt": True
    },
    "Shanghai": {
        "name": "Shanghai", 
        "name_chn": "上海市",
        "districts": ["崇明区", "松江区", "嘉定区", "闵行区", "宝山区", "杨浦区", "奉贤区", 
                     "黄浦区", "徐汇区", "浦东新区", "静安区", "虹口区", "金山区", "青浦区", "普陀区", "长宁区"],
        "language": "chinese",
        "dataset": "Gaode", 
        "expertise": "Shanghai geography expert",
        "use_chinese_prompt": True
    },
    "Singapore": {
        "name": "Singapore",
        "name_chn": "Singapore",
        "districts": [
            "Ang Mo Kio", "Bedok", "Bishan", "Boon Lay", "Bukit Batok", "Bukit Merah",
            "Bukit Panjang", "Bukit Timah", "Central Water Catchment", "Changi",
            "Changi Bay", "Choa Chu Kang", "Clementi", "Downtown Core", "Geylang",
            "Hougang", "Jurong East", "Jurong West", "Kallang", "Marine Parade",
            "Museum", "Newton", "North-Eastern Islands", "Novena", "Orchard",
            "Outram", "Pasir Ris", "Punggol", "Queenstown", "River Valley",
            "Rochor", "Sembawang", "Sengkang", "Serangoon", "Singapore River",
            "Southern Islands", "Sungei Kadut", "Tampines", "Tanglin", "Toa Payoh",
            "Tuas", "Western Islands", "Western Water Catchment", "Woodlands", "Yishun",
            "Lim Chu Kang", "Mandai", "Marina East", "Marina South", "Straits View",
            "Pioneer", "Tengah", "Paya Lebar", "Simpang", "Bukit Timah"
        ],
        "language": "english",
        "dataset": "OSM",
        "district_type": "planning areas",
        "expertise": "Singapore geography expert",
        "use_chinese_prompt": False,
        "features": [
            "Notable landmarks, buildings, or attractions",
            "Shopping centers, markets, or commercial areas", 
            "Transportation hubs or infrastructure",
            "Cultural or recreational facilities",
            "Residential developments or housing estates",
            "Local characteristics or features"
        ]
    },
    "NYC": {
        "name": "NYC",
        "name_chn": "NYC",
        "districts": ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"],
        "language": "english",
        "dataset": "OSM",
        "district_type": "boroughs",
        "expertise": "NYC geography and borough expert",
        "use_chinese_prompt": False,
        "features": [
            "Notable landmarks, buildings, or attractions (e.g., museums, parks, iconic buildings)",
            "Shopping centers, markets, or commercial districts",
            "Transportation hubs (subway stations, bridges, major streets)",
            "Cultural institutions or entertainment venues",
            "Residential developments, housing projects, or neighborhood characteristics",
            "Local businesses, restaurants, or community features",
            "Historical sites or points of interest",
            "Major neighborhoods or districts within the borough"
        ]
    }
}

class UnifiedKeywordGenerator:
    def __init__(self, api_key: str, city_config: Dict, template_type: str = "keywords_kmeans", batch_size: int = 5):
        """Initialize with OpenAI API key, city configuration and processing settings"""
        self.client = OpenAI(api_key=api_key)
        self.city_config = city_config
        self.template_type = template_type
        self.batch_size = batch_size
        self.request_count = 0
        self.start_time = time.time()
        
        # Set up templates
        self.templates = {
            "keywords_kmeans": {
                "chinese": "请根据你对[CITY][REGION]的了解，列出与该地区主要区域功能高度相关的 50 个关键词。请严格按照以下格式输出，不要添加任何解释性文字：'关键词1','关键词2','关键词3',...,'关键词50'。关键词可以是地名、设施、功能形容词等等。",
                "english": "Generate exactly 10 representative keywords for [REGION] in [CITY]. Provide keywords that represent notable landmarks, shopping centers, transportation, cultural facilities, residential areas, and local characteristics. Format: 'keyword1','keyword2',...,'keyword10'"
            }
        }
    
    def check_rate_limit(self):
        """Check and enforce rate limiting"""
        self.request_count += 1
        
        # More conservative rate limiting for mixed usage
        if self.request_count % 2 == 0:
            elapsed = time.time() - self.start_time
            if elapsed < 60:  # Less than 1 minute
                sleep_time = 60 - elapsed + 1  # Add 1 second buffer
                print(f"Rate limiting: waiting {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                self.start_time = time.time()
    
    def postprocess_response_chinese(self, region: str, response: str) -> str:
        """Post-process Chinese response (for Beijing/Shanghai)"""
        if response.startswith(region + "\t"):
            return response

        response = response.strip()
        
        # Try to extract keywords in quotes
        keyword_pattern = r"'([^']+)'"
        keywords = re.findall(keyword_pattern, response)
        
        if keywords:
            formatted_keywords = "'" + "','".join(keywords) + "'"
            return f"{region}\t{formatted_keywords}"
        else:
            # Fallback: try to clean up the response
            cleaned = re.sub(r'^.*?以下是.*?关键词[：:]?\s*', '', response)
            cleaned = re.sub(r'这些关键词.*$', '', cleaned)
            cleaned = re.sub(r'\d+\.\s*', '', cleaned)  # Remove numbering
            
            # Replace various separators with English comma
            cleaned = cleaned.replace('、', ',').replace('，', ',').replace('；', ',').replace(';', ',')
            
            # Split by comma and clean each keyword
            keywords_list = [kw.strip().strip("'\"") for kw in cleaned.split(',') if kw.strip()]
            keywords_list = [kw for kw in keywords_list if kw and len(kw) > 0]
            
            if keywords_list:
                formatted_keywords = "'" + "','".join(keywords_list) + "'"
                return f"{region}\t{formatted_keywords}"
            else:
                return f"{region}\t{response}"
    
    def postprocess_response_english(self, region: str, response: str) -> str:
        """Post-process English response (for Singapore/NYC)"""
        if response.startswith(region + "\t"):
            return response
        
        response = response.strip()
        
        # Extract keywords from various formats
        keyword_pattern = r"'([^']+)'"
        keywords = re.findall(keyword_pattern, response)
        
        if not keywords:
            # Try extracting from numbered lists or plain text
            lines = response.split('\n')
            keywords = []
            for line in lines:
                line = line.strip()
                if line:
                    # Remove numbering and clean
                    cleaned = re.sub(r'^\d+\.\s*', '', line)
                    cleaned = re.sub(r'^-\s*', '', cleaned)
                    cleaned = cleaned.strip().strip("'\"")
                    if cleaned and cleaned not in keywords:
                        keywords.append(cleaned)
        
        if keywords:
            # Ensure proper count based on template
            target_count = 10 if self.city_config["language"] == "english" else 50
            if len(keywords) > target_count:
                keywords = keywords[:target_count]
            elif len(keywords) < target_count:
                # Pad with generic keywords if needed
                while len(keywords) < target_count:
                    keywords.append(f"{region} Area {len(keywords) + 1}")
            
            formatted_keywords = "'" + "','".join(keywords) + "'"
            return f"{region}\t{formatted_keywords}"
        else:
            return f"{region}\t{response}"
    
    def get_prompt_for_district(self, district: str) -> str:
        """Generate prompt for a specific district"""
        if self.city_config["use_chinese_prompt"]:
            template = self.templates[self.template_type]["chinese"]
            city_name = self.city_config["name_chn"]
        else:
            template = self.templates[self.template_type]["english"]
            city_name = self.city_config["name"]
        
        # 强制NYC所有prompt中都用NYC
        if city_name == "NYC":
            city_name = "NYC"
        prompt_template = GenerationTemplate(template)
        return prompt_template.fill(city=city_name, region=district)
    
    def query_gpt_for_district(self, district: str) -> str:
        """Query GPT for a single district"""
        prompt = self.get_prompt_for_district(district)
        
        try:
            self.check_rate_limit()
            
            response = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": f"You are a {self.city_config['expertise']}."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            if "rate_limit_exceeded" in str(e).lower() or "rate limit" in str(e).lower():
                print("Rate limit reached, waiting for ~20 seconds...")
                time.sleep(21)
                return self.query_gpt_for_district(district)  # Retry
            else:
                raise e
    
    def get_keywords_batch(self, districts: List[str]) -> Dict[str, str]:
        """Get keywords for multiple districts in one API call (for English cities)"""
        if self.city_config["use_chinese_prompt"]:
            # Chinese cities use individual processing
            return {}
        
        districts_text = ", ".join(districts)
        features_text = "\n- ".join(self.city_config.get("features", []))
        district_type = self.city_config.get("district_type", "district")
        
        prompt = f"""Generate exactly 10 representative keywords for each of the following {self.city_config["name"]} {district_type}: {districts_text}

For each {district_type[:-1] if district_type.endswith('s') else district_type}, provide keywords that represent:
- {features_text}

Format your response exactly as follows:
DISTRICT_NAME_1:
keyword1
keyword2
...
keyword10

DISTRICT_NAME_2:
keyword1
keyword2
...
keyword10

Provide exactly 10 keywords per {district_type[:-1] if district_type.endswith('s') else district_type}, one per line."""
        
        try:
            self.check_rate_limit()
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a {self.city_config['expertise']}. Provide exactly 10 keywords for each {district_type[:-1] if district_type.endswith('s') else district_type} in the specified format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800 * len(districts),
                temperature=0.3
            )
            
            return self.parse_batch_response(response.choices[0].message.content, districts)
            
        except Exception as e:
            print(f"Error in batch request for {districts}: {str(e)}")
            return {}
    
    def parse_batch_response(self, content: str, districts: List[str]) -> Dict[str, str]:
        """Parse batch response into district -> formatted response mapping"""
        result = {}
        lines = content.strip().split('\n')
        current_district = None
        current_keywords = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line is a district name
            if line.endswith(':') and any(district in line for district in districts):
                if current_district and current_keywords:
                    # Ensure exactly 10 keywords
                    if len(current_keywords) < 10:
                        current_keywords.extend([f"{current_district} Area"] * (10 - len(current_keywords)))
                    formatted_keywords = "'" + "','".join(current_keywords[:10]) + "'"
                    result[current_district] = f"{current_district}\t{formatted_keywords}"
                
                current_district = line.replace(':', '').strip()
                current_keywords = []
            else:
                # This is a keyword
                if current_district:
                    current_keywords.append(line)
        
        # Handle the last district
        if current_district and current_keywords:
            if len(current_keywords) < 10:
                current_keywords.extend([f"{current_district} Area"] * (10 - len(current_keywords)))
            formatted_keywords = "'" + "','".join(current_keywords[:10]) + "'"
            result[current_district] = f"{current_district}\t{formatted_keywords}"
        
        return result
    
    def save_to_file(self, answer: str, file_path: str):
        """Save response to file"""
        with open(file_path, "a", encoding="utf-8") as file:
            file.write(f"{answer}\n")
    
    def get_output_file_path(self) -> str:
        """Get output file path based on city configuration"""
        suburban_dir = get_suburban_dir()
        dataset = self.city_config["dataset"]
        city_name = self.city_config["name"]
        # 强制NYC输出路径和文件夹名都用NYC
        if city_name == "NYC":
            city_name = "NYC"
        file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city_name, f'district_desc_{self.template_type}.txt')
        return file_path
    
    def generate_all_keywords(self, use_batch: bool = True):
        """Generate keywords for all districts and save to file"""
        districts = self.city_config["districts"]
        file_path = self.get_output_file_path()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Clean the contents in existing file
        with open(file_path, "w", encoding="utf-8") as file:
            pass
        
        print(f"Generating keywords for {len(districts)} {self.city_config['name']} districts...")
        print(f"Output file: {file_path}")
        
        # Choose processing method based on city language and user preference
        if self.city_config["use_chinese_prompt"] or not use_batch:
            # Individual processing for Chinese cities or when batch is disabled
            self.process_individually(districts, file_path)
        else:
            # Batch processing for English cities
            self.process_in_batches(districts, file_path)
    
    def process_individually(self, districts: List[str], file_path: str):
        """Process districts individually (used for Chinese cities or fallback)"""
        for district in tqdm(districts, desc="Generating descriptions"):
            while True:
                try:
                    response = self.query_gpt_for_district(district)
                    
                    # Post-process based on language
                    if self.city_config["use_chinese_prompt"]:
                        processed_response = self.postprocess_response_chinese(district, response)
                    else:
                        processed_response = self.postprocess_response_english(district, response)
                    
                    self.save_to_file(processed_response, file_path)
                    time.sleep(3)
                    break
                    
                except Exception as e:
                    if "rate_limit_exceeded" in str(e).lower() or "rate limit" in str(e).lower():
                        print("Rate limit reached, waiting for ~20 seconds...")
                        time.sleep(21)
                    else:
                        print(f"Error occurred: {e}")
                        print("Retrying in 5 seconds...")
                        time.sleep(5)
    
    def process_in_batches(self, districts: List[str], file_path: str):
        """Process districts in batches (used for English cities)"""
        print(f"Using batch processing with batch size: {self.batch_size}")
        
        for i in tqdm(range(0, len(districts), self.batch_size), desc="Processing batches"):
            batch = districts[i:i + self.batch_size]
            
            while True:
                try:
                    batch_results = self.get_keywords_batch(batch)
                    
                    if batch_results:
                        # Save batch results
                        for district in batch:
                            if district in batch_results:
                                self.save_to_file(batch_results[district], file_path)
                            else:
                                # Fallback to individual processing
                                response = self.query_gpt_for_district(district)
                                processed_response = self.postprocess_response_english(district, response)
                                self.save_to_file(processed_response, file_path)
                    else:
                        # Batch failed, process individually
                        for district in batch:
                            response = self.query_gpt_for_district(district)
                            processed_response = self.postprocess_response_english(district, response)
                            self.save_to_file(processed_response, file_path)
                            time.sleep(2)
                    
                    # Add delay between batches
                    if i + self.batch_size < len(districts):
                        time.sleep(3)
                    break
                    
                except Exception as e:
                    if "rate_limit_exceeded" in str(e).lower() or "rate limit" in str(e).lower():
                        print("Rate limit reached, waiting for ~20 seconds...")
                        time.sleep(21)
                    else:
                        print(f"Error occurred: {e}")
                        print("Retrying in 5 seconds...")
                        time.sleep(5)

def get_available_cities() -> str:
    """Get formatted string of available cities"""
    cities = list(CITY_CONFIGS.keys())
    return ", ".join(cities)

def validate_city(city: str) -> Tuple[bool, str]:
    """Validate if the city is supported"""
    # Create a mapping from lowercase to proper case
    city_mapping = {key.lower(): key for key in CITY_CONFIGS.keys()}
    
    if city.lower() not in city_mapping:
        available = get_available_cities()
        return False, f"City '{city}' is not supported. Available cities: {available}"
    return True, city_mapping[city.lower()]

def main():
    parser = argparse.ArgumentParser(
        description="Generate keywords for city districts using OpenAI API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available cities: Beijing, Shanghai, Singapore, NYC

Examples:
  python script.py --city Beijing
  python script.py --city Singapore --individual
  python script.py --city NYC --batch-size 3
        """
    )
    
    parser.add_argument(
        '--city', 
        type=str, 
        default='Beijing',
        help=f"City to generate keywords for. Available: Beijing, Shanghai, Singapore, NYC"
    )
    
    parser.add_argument(
        '--template', 
        type=str, 
        default='keywords_kmeans',
        help="Template type: 'keywords_kmeans'"
    )
    
    parser.add_argument(
        '--dataset', 
        type=str,
        help="Override dataset (default: auto-detect based on city)"
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        default="insert-your-api-key",
        help="OpenAI API key"
    )
    
    parser.add_argument(
        '--individual',
        action="store_true",
        help="Force individual processing instead of batch processing (for English cities)"
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=5,
        help="Batch size for batch processing (default: 5, only for English cities)"
    )
    
    args = parser.parse_args()
    
    # Validate city
    is_valid, result = validate_city(args.city)
    if not is_valid:
        print(f"Error: {result}")
        return 1
    
    # Get the proper case city name
    proper_city_name = result
    
    # Get city configuration
    city_config = CITY_CONFIGS[proper_city_name].copy()
    
    # Override dataset if specified
    if args.dataset:
        city_config["dataset"] = args.dataset
    
    # Display configuration
    print(f"City: {city_config['name']}")
    print(f"Districts: {len(city_config['districts'])}")
    print(f"Language: {city_config['language']}")
    print(f"Dataset: {city_config['dataset']}")
    
    use_batch = not args.individual and not city_config["use_chinese_prompt"]
    
    if use_batch:
        print(f"Processing method: Batch (size: {args.batch_size})")
    else:
        print("Processing method: Individual")
        if city_config["use_chinese_prompt"]:
            print("(Chinese cities always use individual processing)")
    
    # Generate keywords
    generator = UnifiedKeywordGenerator(
        api_key=args.api_key,
        city_config=city_config,
        template_type=args.template,
        batch_size=args.batch_size
    )
    
    generator.generate_all_keywords(use_batch=use_batch)
    print(f"\n✓ Successfully completed keyword generation for {city_config['name']}")

if __name__ == "__main__":
    main()