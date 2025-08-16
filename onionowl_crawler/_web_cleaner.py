import asyncio
from collections import Counter
from playwright.async_api import async_playwright
from lxml import html as lxml_html, etree

PROXY_SERVER = "socks5://127.0.0.1:9050"
REMOVE_TAGS = ['script', 'style', 'meta', 'link', 'iframe', 'noscript', 'nav', 'header', 'footer']
AD_KEYWORDS = ['ad', 'ads', 'sponsor', 'tracked', 'banner', 'cookie', 'popup']

def remove_unwanted_inside(element):
    for tag in REMOVE_TAGS:
        for el in element.xpath(f'.//{tag}'):
            el.drop_tree()
    for keyword in AD_KEYWORDS:
        for el in element.xpath(
            f'.//*[contains(translate(@class, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "{keyword}") '
            f'or contains(translate(@id, "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"), "{keyword}")]'
        ):
            el.drop_tree()

def remove_empty_tags(element):
    # Remove empty tags (except for tags that can have meaningful whitespace/tails)
    for el in list(element):
        remove_empty_tags(el)
    for el in list(element):
        has_no_children = len(el) == 0
        has_no_text = (el.text is None) or (el.text.strip() == '')
        has_no_attrs = not el.attrib
        # Don't remove <br> and <img> even if "empty"
        if has_no_children and has_no_text and has_no_attrs and el.tag not in ('br', 'img'):
            element.remove(el)

def minify_html(html):
    html = html.replace('\n', '').replace('\r', '')
    while '> <' in html or '>  <' in html:
        html = html.replace('> <', '><').replace('>  <', '><')
    html = ' '.join(html.split())
    return html.strip()

def find_main_divs(tree):
    body = tree.find('.//body')
    if body is None:
        return []
    # Find all div class candidates
    class_cands = []
    for el in body.xpath('.//div[@class]'):
        classes = el.attrib['class'].strip().split()
        for c in classes:
            class_cands.append(c)
    counter = Counter(class_cands)
    main_classes = [cls for cls, count in counter.items() if count >= 2]
    divs = []
    seen = set()
    if main_classes:
        for card_class in main_classes:
            for el in body.xpath(f'.//div[contains(concat(" ", normalize-space(@class), " "), " {card_class} ")]'):
                if id(el) not in seen:
                    divs.append(el)
                    seen.add(id(el))
        return divs
    # Fallback: all direct <section>, <article>, <div> under body
    divs = body.xpath('./section | ./article | ./div')
    if divs:
        return divs
    return [body]  # last resort

def get_minified_chunks(raw_html):
    parser = lxml_html.HTMLParser(encoding='utf-8')
    tree = lxml_html.fromstring(raw_html, parser=parser)
    divs = find_main_divs(tree)
    chunks = []
    for el in divs:
        # Clean in place: DO NOT reparse!
        remove_unwanted_inside(el)
        remove_empty_tags(el)
        chunk_html = etree.tostring(el, encoding='unicode', method='html', with_tail=False)
        minified = minify_html(chunk_html)
        if minified:
            chunks.append(minified)
    return chunks

async def scrape_and_clean(url, output_txt="scraped_cleaned.txt"):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, proxy={"server": PROXY_SERVER})
        page = await browser.new_page()
        await page.goto(url, timeout=90000)
        html_content = await page.content()
        await browser.close()

    html_chunks = get_minified_chunks(html_content)

    with open(output_txt, "w", encoding="utf-8") as f:
        for chunk in html_chunks:
            f.write('{"' + chunk.replace('"', '\\"') + '"}\n')

    print(f"Saved cleaned data to {output_txt}")

if __name__ == "__main__":
    url = input("Enter the website URL to scrape and clean (supports .onion via Tor): ").strip()
    output_txt = input("Enter output TXT file name (default: scraped_cleaned.txt): ").strip()
    if not output_txt:
        output_txt = "scraped_cleaned.txt"
    asyncio.run(scrape_and_clean(url, output_txt))