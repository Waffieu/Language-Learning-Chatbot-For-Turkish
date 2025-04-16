import logging
import sys
from word_translation import identify_uncommon_words, translate_words, process_text_with_translations

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_translation.log')
    ]
)
logger = logging.getLogger(__name__)

def test_translation():
    # Test sentences in different languages
    test_cases = [
        # German test cases
        {
            "text": "Ich bin hart satz und ich bin aufmerksamkeit und übermensch und ich liebe fuchs und sie sind süss",
            "language": "German"
        },
        {
            "text": "Ja, ich weiß Heldmaschine! Wir haben schon oft darüber gesprochen!",
            "language": "German"
        },
        # User's specific test case
        {
            "text": "Heldmaschine ist eine deutsche Band!\n\nSie machen Musik, die NDH heißt.\n\nIch habe im Internet nachgeschaut.\n\n*   Sie wurde 2008 gegründet. 👍\n*   Sie kommt aus Koblenz. Magst du ihre Musik? 🦊🔧",
            "language": "German"
        },
        # Test case with multiple sentences containing uncommon words
        {
            "text": "Die Quantenphysik ist ein faszinierendes Gebiet der Wissenschaft. Die Thermodynamik untersucht die Beziehung zwischen Wärme und anderen Formen der Energie.",
            "language": "German"
        },
        # English test case
        {
            "text": "I understand the concept of quantum physics and thermodynamics.",
            "language": "English"
        },
        # French test case
        {
            "text": "J'aime la philosophie et l'astrophysique. C'est très intéressant.",
            "language": "French"
        },
        # Spanish test case
        {
            "text": "Me gusta la arquitectura y la neurociencia. Es muy interesante.",
            "language": "Spanish"
        }
    ]

    # Print header
    print("\n\n" + "*"*80)
    print("TRANSLATION TEST RESULTS")
    print("*"*80)

    # Test each case
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        language = test_case["language"]

        print(f"\n\n===== Test Case {i+1}: {language} =====")
        print(f"Original: {text}")

        # Step 1: Identify uncommon words
        uncommon_words = identify_uncommon_words(text, language)
        print(f"\nIdentified uncommon words: {uncommon_words}")

        # Step 2: Translate uncommon words
        if uncommon_words:
            translations = translate_words(uncommon_words, language)
            print(f"\nTranslations: {translations}")
        else:
            print("\nNo uncommon words to translate.")

        # Step 3: Process the full text with translations
        processed_text = process_text_with_translations(text, language)
        print(f"\nProcessed text:\n{processed_text}")

        print("\n" + "="*80)

    # Print summary
    print("\n\nTEST COMPLETED SUCCESSFULLY")
    print("*"*80)

if __name__ == "__main__":
    test_translation()
