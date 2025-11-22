import os
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Analyzer:
    def run(self, topic: str):
        # simple prompt to summarize the topic in 2 sentences
        prompt = f"Résume en deux phrases le sujet suivant pour guider la rédaction d'un essai scientifique:\n\nSujet: {topic}"
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=150
        )
        return resp.choices[0].message.content
