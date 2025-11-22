import os
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Planner:
    def run(self, analysis: str, sources=None):
        prompt = f"En te basant sur cette analyse:\n{analysis}\nGénère un plan pour un essai scientifique: introduction, 2-3 parties, conclusion. Indique ~nombre de mots par section."
        if sources:
            prompt += "\nVoici quelques sources utiles:\n" + "\n".join([s.get('title','') + ' - ' + s.get('url','') for s in sources])
        resp = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=[{'role':'user','content':prompt}],
            max_tokens=300
        )
        return resp.choices[0].message.content
