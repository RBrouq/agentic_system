import os
import requests

class WebSearcher:
    def __init__(self):
        self.api_key = os.environ.get('TAVILY_API_KEY')
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY n'est pas définie dans l'environnement")
        self.endpoint = "https://api.tavily.com/search"  # à adapter selon doc TAVILY

    def search(self, query, num=3):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"q": query, "limit": num}
        try:
            resp = requests.get(self.endpoint, headers=headers, params=params, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", [])[:num]:
                results.append({
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("snippet")
                })
            return results
        except Exception as e:
            print("Erreur recherche TAVILY:", e)
            return []


