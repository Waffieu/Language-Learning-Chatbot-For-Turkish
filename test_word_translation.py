import logging
from word_translation import identify_uncommon_words, translate_words, process_text_with_translations

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def test_word_translation():
    """Test that word translation works correctly"""

    # Test languages
    languages = ["English", "Spanish", "German"]

    # Test sentences
    test_sentences = [
        "The algorithm efficiently processes complex data structures.",
        "Quantum computing leverages superposition and entanglement to perform calculations.",
        "The architecture of the system enables scalability and resilience.",
        "La inteligencia artificial está transformando muchas industrias.",
        "Der Algorithmus verarbeitet komplexe Datenstrukturen effizient."
    ]

    print("Testing word translation:")
    print("==========================")

    for language, sentence in zip(languages, test_sentences):
        print(f"\nLanguage: {language}")
        print(f"Sentence: {sentence}")
        print("-" * 50)

        # Identify uncommon words
        uncommon_words = identify_uncommon_words(sentence, language)
        print(f"Uncommon words: {uncommon_words}")

        # Translate uncommon words to Turkish
        if uncommon_words:
            translations = translate_words(uncommon_words, language)
            print("\nTranslations:")
            for word, translation in translations.items():
                print(f"  {word}: {translation}")

        # Process the entire text
        processed_text = process_text_with_translations(sentence, language)
        print(f"\nProcessed text:\n{processed_text}")
        print("-" * 50)

    # Test a longer text with multiple sentences
    print("\nTesting multi-sentence text:")
    print("============================")

    multi_sentence_text = """
    Artificial intelligence is revolutionizing many industries. Machine learning algorithms can analyze vast amounts of data to identify patterns. 
    Neural networks are particularly effective for image recognition tasks. The computational requirements for training these models are substantial.
    """

    print(f"\nOriginal text:\n{multi_sentence_text}")
    processed_multi = process_text_with_translations(multi_sentence_text, "English")
    print(f"\nProcessed text:\n{processed_multi}")

    print("\nTest completed!")

if __name__ == "__main__":
    test_word_translation()
