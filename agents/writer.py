import os
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Writer:
    def run(self, plan: str, sources=None, hints: str = ""):
        prompt = f"Rédige un essai en suivant ce plan:\n{plan}\n"
        if sources:
            prompt += "\nIntègre des références et citations basées sur ces sources:\n"
            for s in sources:
                prompt += f"- {s.get('title','')} ({s.get('url','')})\n"
        if hints:
            prompt += "\nCorrige selon ces indications:\n" + hints
        prompt += "\nLongueur totale ~600-900 mots. Style: académique simple, français."
        resp = client.chat.completions.create(
            model='gpt-4.1',
            messages=[{'role':'user','content':prompt}],
            max_tokens=1200
        )
        return resp.choices[0].message.content
