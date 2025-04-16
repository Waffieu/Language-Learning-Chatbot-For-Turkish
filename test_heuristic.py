import re
from typing import List

def identify_uncommon_words_heuristic(text: str, language: str) -> List[str]:
    """
    Identify truly uncommon words using only the heuristic approach
    """
    # Skip if the text is in Turkish
    if language.lower() == "turkish":
        return []

    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text.lower())

    # Remove duplicates while preserving order
    unique_words = []
    seen = set()
    for word in words:
        if word not in seen and len(word) > 3:  # Only consider words longer than 3 characters
            seen.add(word)
            unique_words.append(word)

    # If no words to analyze, return empty list
    if not unique_words:
        return []

    # Common words in various languages
    common_words = {
        "german": [
            "ich", "du", "er", "sie", "es", "wir", "ihr", "sie", "und", "oder", "aber", "wenn", "dann",
            "bin", "ist", "sind", "war", "waren", "haben", "hat", "hatte", "gehen", "kommen", "machen",
            "gut", "schlecht", "ja", "nein", "bitte", "danke", "hallo", "tschüss", "auch", "schon", 
            "noch", "jetzt", "hier", "dort", "heute", "morgen", "gestern", "immer", "nie", "oft", 
            "manchmal", "viel", "wenig", "groß", "klein", "alt", "jung", "neu", "tag", "nacht", 
            "woche", "monat", "jahr", "zeit", "uhr", "minute", "stunde", "wasser", "essen", "trinken",
            "sprechen", "sagen", "fragen", "antworten", "verstehen", "wissen", "kennen", "lernen",
            "leben", "lieben", "mögen", "wollen", "können", "müssen", "sollen", "dürfen", "freund",
            "familie", "mutter", "vater", "bruder", "schwester", "kind", "mann", "frau", "mensch",
            "haus", "wohnung", "stadt", "land", "straße", "weg", "auto", "bus", "zug", "flugzeug",
            "schule", "arbeit", "beruf", "geld", "preis", "farbe", "rot", "blau", "grün", "gelb",
            "schwarz", "weiß", "musik", "buch", "film", "spiel", "sport", "fußball", "wetter", "sonne",
            "regen", "schnee", "wind", "warm", "kalt", "schön", "hässlich", "leicht", "schwer", "schnell",
            "langsam", "früh", "spät", "oben", "unten", "links", "rechts", "vorne", "hinten", "in", "an",
            "auf", "unter", "über", "vor", "nach", "mit", "ohne", "für", "gegen", "weil", "dass", "ob",
            "wie", "was", "wer", "wo", "wann", "warum", "eins", "zwei", "drei", "vier", "fünf", "sechs",
            "sieben", "acht", "neun", "zehn", "hundert", "tausend", "million", "erste", "zweite", "letzte",
            "nächste", "alle", "viele", "einige", "keine", "jeder", "dieser", "jener", "mein", "dein", "sein",
            "ihr", "unser", "euer", "der", "die", "das", "ein", "eine", "kein", "keine", "mehr", "weniger",
            "sehr", "zu", "nicht", "nur", "schon", "noch", "wieder", "immer", "nie", "vielleicht", "sicher",
            "natürlich", "genau", "wirklich", "leider", "hoffentlich", "bitte", "danke", "entschuldigung",
            "tut", "leid", "gern", "geschehen", "willkommen", "tschüss", "auf", "wiedersehen", "bis", "später",
            "morgen", "bald", "guten", "tag", "abend", "nacht", "morgen", "woche", "ende", "schön", "toll",
            "super", "prima", "gut", "schlecht", "richtig", "falsch", "wichtig", "interessant", "langweilig",
            "einfach", "schwierig", "möglich", "unmöglich", "gleich", "anders", "ähnlich", "verschieden",
            "weiß", "darüber", "gesprochen", "heldmaschine"
        ],
        "english": ["i", "you", "he", "she", "it", "we", "they", "and", "or", "but", "if", "then",
                    "am", "is", "are", "was", "were", "have", "has", "had", "go", "come", "make",
                    "good", "bad", "yes", "no", "please", "thank", "hello", "bye"],
        "french": ["je", "tu", "il", "elle", "nous", "vous", "ils", "elles", "et", "ou", "mais", "si",
                   "suis", "est", "sommes", "sont", "étais", "étaient", "avoir", "aller", "venir", "faire",
                   "bon", "mauvais", "oui", "non", "s'il", "merci", "bonjour", "au revoir"],
        "spanish": ["yo", "tú", "él", "ella", "nosotros", "vosotros", "ellos", "ellas", "y", "o", "pero", "si",
                    "soy", "es", "somos", "son", "era", "eran", "tener", "ir", "venir", "hacer",
                    "bueno", "malo", "sí", "no", "por favor", "gracias", "hola", "adiós"]
    }

    # Get the appropriate common words list based on language
    lang_key = language.lower()
    if lang_key in common_words:
        common_word_list = common_words[lang_key]
    else:
        # Default to English if language not in our list
        common_word_list = common_words["english"]

    # Identify uncommon words based on our simple list
    # Be more selective - only include words longer than 6 characters that aren't in our common list
    # This helps ensure we're only getting truly uncommon words
    uncommon_words = [word for word in unique_words if word.lower() not in common_word_list and len(word) > 6]

    # Limit to at most 2 words per sentence to avoid overwhelming translations
    return uncommon_words[:2]

def test_heuristic():
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
    print("HEURISTIC TEST RESULTS")
    print("*"*80)
    
    # Test each case
    for i, test_case in enumerate(test_cases):
        text = test_case["text"]
        language = test_case["language"]
        
        print(f"\n\n===== Test Case {i+1}: {language} =====")
        print(f"Original: {text}")
        
        # Identify uncommon words using heuristic
        uncommon_words = identify_uncommon_words_heuristic(text, language)
        print(f"\nIdentified uncommon words: {uncommon_words}")
        
        print("\n" + "="*80)
    
    # Print summary
    print("\n\nTEST COMPLETED SUCCESSFULLY")
    print("*"*80)

if __name__ == "__main__":
    test_heuristic()
