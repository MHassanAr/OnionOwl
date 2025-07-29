import requests

proxies = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050',
}

url = "http://www.xyz.onion"

try:
    response = requests.get(url, proxies=proxies, timeout=30)
    if response.status_code == 200:
        print("Response from .onion site:")
        print(response.text)
    else:
        print(f"Failed to connect. HTTP Status Code: {response.status_code}")
except requests.exceptions.RequestException as e:
    print("An error occurred:", e)