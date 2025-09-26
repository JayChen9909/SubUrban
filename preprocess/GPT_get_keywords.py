from openai import OpenAI
import time
from tqdm import tqdm
import argparse
import os

def get_suburban_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)  # Go from preprocess folder to SubUrban root

class GenerationTemplate:
    def __init__(self, template):
        self.template = template

    def fill(self, city='', region=''):
        return self.template.replace('[CITY]', city).replace('[REGION]', region)


def postprocess_response(region, response):
    if response.startswith(region + "\t"):
        return response

    response = response.strip()
    
    # Remove any extra explanatory text and keep only the keywords
    # Look for patterns like '关键词1','关键词2',... or similar
    import re
    
    # Try to extract keywords in quotes
    keyword_pattern = r"'([^']+)'"
    keywords = re.findall(keyword_pattern, response)
    
    if keywords:
        # Format as required: 'keyword1','keyword2','keyword3',... with English commas
        formatted_keywords = "'" + "','".join(keywords) + "'"
        return f"{region}\t{formatted_keywords}"
    else:
        # Fallback: try to clean up the response
        # Remove common unwanted phrases and formatting
        cleaned = re.sub(r'^.*?以下是.*?关键词[：:]?\s*', '', response)
        cleaned = re.sub(r'这些关键词.*$', '', cleaned)
        cleaned = re.sub(r'\d+\.\s*', '', cleaned)  # Remove numbering
        
        # Replace various separators with English comma
        cleaned = cleaned.replace('、', ',').replace('，', ',').replace('；', ',').replace(';', ',')
        
        # Split by comma and clean each keyword
        keywords_list = [kw.strip().strip("'\"") for kw in cleaned.split(',') if kw.strip()]
        
        # Filter out empty keywords and format properly
        keywords_list = [kw for kw in keywords_list if kw and len(kw) > 0]
        
        if keywords_list:
            # Format as required: 'keyword1','keyword2','keyword3',... with English commas
            formatted_keywords = "'" + "','".join(keywords_list) + "'"
            return f"{region}\t{formatted_keywords}"
        else:
            # Last resort: just return the cleaned response with tab separator
            return f"{region}\t{response}"

def query_gpt_for_region_description(region, city, prompt_template, client):
    city_chn = "上海市" if city == "Shanghai" else "北京市"
    prompt = prompt_template.fill(city=city_chn, region=region)

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        temperature=0.7,
    )
    return response.choices[0].message.content


def save_to_file(answer, file_path):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"{answer}\n")


def main():
    client = OpenAI(
        api_key='insert_your_api_key_here'  # Replace with your actual API key
    )

    parser = argparse.ArgumentParser()
    # parser.add_argument('--version', type=str, default='v2', help="版本号，用于生成文件名")
    parser.add_argument('--city', type=str, default='Beijing', help="City: 'Shanghai' or 'Beijing'")
    parser.add_argument('--template', type=str, default='keywords_kmeans', help="Template type: 'keywords_kmeans'")
    parser.add_argument('--dataset', type=str, default='Gaode', help="Datasets from Meituan, Gaode")
    args = parser.parse_args()

    # version = args.version
    city = args.city
    template_type = args.template
    dataset = args.dataset

    if city == "Shanghai":
        region_pool = ["崇明区", "松江区", "嘉定区", "闵行区", "宝山区", "杨浦区", "奉贤区", 
                       "黄浦区", "徐汇区", "浦东新区", "静安区", "虹口区", "金山区", "青浦区", "普陀区", "长宁区"]
    elif city == "Beijing":
        region_pool = ["东城区", "西城区", "朝阳区", "海淀区", "丰台区", "石景山区", "通州区", 
                       "昌平区", "大兴区", "顺义区", "房山区", "门头沟区", "平谷区", "怀柔区", "密云区", "延庆区"]

    templates = {
        "keywords_kmeans": "请根据你对[CITY][REGION]的了解，列出与该地区主要区域功能高度相关的 50 个关键词。请严格按照以下格式输出，不要添加任何解释性文字：'关键词1','关键词2','关键词3',...,'关键词50'。关键词可以是地名、设施、功能形容词等等。"
    }

    if template_type not in templates:
        raise ValueError(f"Unsupported template type: {template_type}. Supported types are: {list(templates.keys())}")

    prompt_template = GenerationTemplate(templates[template_type])

    suburban_dir = get_suburban_dir()
    file_path = os.path.join(suburban_dir, 'data', dataset, 'projected', city, f'district_desc_{template_type}.txt')
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Clean the contents in existing file
    with open(file_path, "w", encoding="utf-8") as file:
        pass

    for region in tqdm(region_pool, desc="Generating descriptions"):
        while True:
            try:
                region_description = query_gpt_for_region_description(region, city, prompt_template, client)
                region_description = postprocess_response(region, region_description)
                save_to_file(region_description, file_path)
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


if __name__ == "__main__":
    main()
