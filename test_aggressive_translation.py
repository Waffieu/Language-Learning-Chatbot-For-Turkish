import logging
import sys
from word_translation import identify_uncommon_words, translate_words, post_process_response

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Test text with simple sentences that should now have more translations
test_text = """Guten Tag! Ich heiße Max und ich komme aus Deutschland.

Ich lerne Türkisch seit drei Monaten. Es ist eine schöne Sprache, aber manchmal schwierig.

Ich mag Musik hören und Bücher lesen. Mein Lieblingsbuch ist "Der kleine Prinz".

Kannst du mir helfen, mein Türkisch zu verbessern?"""

# Process the text
processed_text = post_process_response(test_text, "German")

# Print the results
print("\n\n=== ORIGINAL TEXT ===")
print(test_text)
print("\n\n=== PROCESSED TEXT WITH MORE AGGRESSIVE TRANSLATIONS ===")
print(processed_text)

# Test with English text
english_text = """Hello! My name is Sarah and I'm from the United States.

I've been learning Turkish for about two months. It's an interesting language with a different structure.

I enjoy traveling and photography. Last year I visited Istanbul and took many pictures.

Can you help me practice my Turkish vocabulary?"""

# Process the English text
processed_english = post_process_response(english_text, "English")

# Print the results
print("\n\n=== ORIGINAL ENGLISH TEXT ===")
print(english_text)
print("\n\n=== PROCESSED ENGLISH TEXT WITH MORE AGGRESSIVE TRANSLATIONS ===")
print(processed_english)
