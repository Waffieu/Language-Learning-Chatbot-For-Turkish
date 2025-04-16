import logging
import sys
from word_translation import post_process_response

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Test text in German (the user's example)
test_text = """Heldmaschine ist eine deutsche Band!

Sie machen Musik, die NDH heißt.

Ich habe im Internet nachgeschaut.

*   Sie wurde 2008 gegründet. 👍
*   Sie kommt aus Koblenz. Magst du ihre Musik? 🦊🔧"""

# Process the text
processed_text = post_process_response(test_text, "German")

# Print the results
print("\n\n=== ORIGINAL TEXT ===")
print(test_text)
print("\n\n=== PROCESSED TEXT ===")
print(processed_text)

# Test with multiple sentences containing uncommon words
test_text2 = """Die Quantenphysik ist ein faszinierendes Gebiet der Wissenschaft. 
Die Thermodynamik untersucht die Beziehung zwischen Wärme und anderen Formen der Energie."""

# Process the text
processed_text2 = post_process_response(test_text2, "German")

# Print the results
print("\n\n=== ORIGINAL TEXT 2 ===")
print(test_text2)
print("\n\n=== PROCESSED TEXT 2 ===")
print(processed_text2)
