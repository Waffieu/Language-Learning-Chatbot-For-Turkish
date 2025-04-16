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

# Test text with existing wort schatz sections in the middle
test_text = """Heldmaschine ist eine deutsche Band!
wort schatz
heldmaschine = kahraman makinesi 😊

Sie machen Musik, die NDH heißt. wort schatz
heißt = denir

Ich habe im Internet nachgeschaut. wort schatz
internet = internet

*   Sie wurde 2008 gegründet. 👍
*   Sie kommt aus Koblenz. Magst du ihre Musik? 🦊🔧"""

# Process the text
processed_text = post_process_response(test_text, "German")

# Print the results
print("\n\n=== ORIGINAL TEXT WITH EXISTING WORT SCHATZ SECTIONS ===")
print(test_text)
print("\n\n=== PROCESSED TEXT (SHOULD HAVE ONLY ONE WORT SCHATZ AT THE END) ===")
print(processed_text)

# Test with multiple sentences containing uncommon words
test_text2 = """Die Quantenphysik ist ein faszinierendes Gebiet der Wissenschaft. 
wort schatz
quantenphysik = kuantum fiziği

Die Thermodynamik untersucht die Beziehung zwischen Wärme und anderen Formen der Energie.
wort schatz
thermodynamik = termodinamik"""

# Process the text
processed_text2 = post_process_response(test_text2, "German")

# Print the results
print("\n\n=== ORIGINAL TEXT 2 WITH EXISTING WORT SCHATZ SECTIONS ===")
print(test_text2)
print("\n\n=== PROCESSED TEXT 2 (SHOULD HAVE ONLY ONE WORT SCHATZ AT THE END) ===")
print(processed_text2)
