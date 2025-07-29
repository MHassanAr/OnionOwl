import csv
import requests
import re
from collections import Counter
from bs4 import BeautifulSoup
import os
from nltk.corpus import stopwords

PROXIES = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050',
}

def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    full_text = ' '.join(soup.stripped_strings)
    clean_text = ' '.join(full_text.split())
    return clean_text

def extract_keywords_with_frequency(text, num_keywords=10):
    stop_words = set(stopwords.words('english'))
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    return word_counts.most_common(num_keywords)

def process_urls(input_csv, output_csv, top_n_keywords=10):
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found!")
        return

    with open(input_csv, 'r', encoding='utf-8-sig') as infile, open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        csv_reader = csv.reader(infile)
        rows = list(csv_reader)
        if not rows:
            print(f"Error: Input file '{input_csv}' is empty.")
            return

        header = rows[0]
        data_rows = rows[1:]

        keyword_headers = []
        for i in range(1, top_n_keywords + 1):
            keyword_headers.append(f'keyword{i}')
            keyword_headers.append(f'freq{i}')
        extended_header = header + keyword_headers

        csv_writer = csv.writer(outfile)
        csv_writer.writerow(extended_header)

        for index, row in enumerate(data_rows, start=1):
            url = row[0].strip()
            print(f"Processing link #{index}: {url}")

            keyword_freq_pairs = ["" for _ in range(top_n_keywords * 2)]

            if url:
                try:
                    response = requests.get(url, proxies=PROXIES, timeout=30)
                    if response.status_code == 200:
                        clean_text = extract_text_from_html(response.text)
                        top_keywords = extract_keywords_with_frequency(clean_text, top_n_keywords)

                        for i, (keyword, freq) in enumerate(top_keywords):
                            keyword_freq_pairs[i * 2] = keyword
                            keyword_freq_pairs[i * 2 + 1] = str(freq)
                    else:
                        print(f"Failed to fetch {url}: HTTP {response.status_code}")
                except requests.exceptions.RequestException:
                    print(f"Error fetching {url}, marking keywords as blank.")

            complete_row = row + keyword_freq_pairs
            csv_writer.writerow(complete_row)

    print(f"\nKeyword extraction completed. Results saved to '{output_csv}'.")

if __name__ == "__main__":
    input_csv = "onionowl_input_data.csv"
    output_csv = "onionowl_output_data.csv"
    process_urls(input_csv, output_csv)