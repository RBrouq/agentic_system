import json, re
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Critic:
    def run(self, text: str):
        # Nettoyage du texte pour éviter accents problématiques
        text_clean = text.replace("’", "'").replace("“", '"').replace("”", '"').strip()

        prompt = (
            "Tu es un critique académique. Évalue le texte ci-dessous et retourne uniquement un JSON "
            "avec ces clés : score(0-100), needs_revision(bool), suggested_changes(string). "
            "Ne réponds jamais avec du texte explicatif.\n"
            f"Texte :\n{text_clean}"
        )

        resp = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role':'user','content':prompt}],
            max_tokens=300
        )

        reply = resp.choices[0].message.content
        # Supprime les balises ```json si présentes
        reply_clean = re.sub(r"```json|```", "", reply).strip()

        try:
            return json.loads(reply_clean)
        except:
            # En cas d'échec, on renvoie le texte brut pour debug
            return {'raw': reply_clean}


