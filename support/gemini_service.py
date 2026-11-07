
from google import genai
from django.conf import settings


def generate_gemini_response(prompt):
    """
    Send a prompt to Google's Gemini model
    and return its text response.
    """

    if not settings.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing. "
            "Add your API key to the .env file."
        )

    client = genai.Client(
        api_key=settings.GEMINI_API_KEY
    )

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt
    )

    return response.text
